# -*- coding: utf-8 -*-
import json
import os
import time
from typing import Optional

import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.tasks import get_task_dict
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM

from module.moe_lora import inject_lora_into_moe, freeze_non_lora_params
from module.Construct_context import construct_context


def get_task_family(task):
    output_type = getattr(task, "OUTPUT_TYPE", None)
    if output_type in ["loglikelihood", "multiple_choice"]:
        return "discriminative"
    if output_type in ["generate_until"]:
        return "generative"
    raise ValueError(f"Unknown task OUTPUT_TYPE: {output_type}")


def get_main_metric(task, task_result):
    if task.OUTPUT_TYPE in ["loglikelihood", "multiple_choice"]:
        return task_result.get("acc,none", 0.0), task_result.get("acc_stderr,none", 0.0)
    if task.OUTPUT_TYPE == "generate_until":
        return task_result.get("exact_match,none", 0.0), task_result.get("exact_match,none", 0.0)
    raise ValueError(f"Unknown OUTPUT_TYPE: {task.OUTPUT_TYPE}")


def _collect_trainable_initial_params(model):
    return {
        name: p.detach().clone().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


class Args:
    def __init__(
        self,
        task,
        model_name: Optional[str] = "./Qwen1.5-MoE-A2.7B-Chat",
        eval_batch_size: Optional[int] = 4,
        result_path: Optional[str] = None,
        lr: Optional[float] = 1e-5,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 16,
        reg_lambda: Optional[float] = 0.0,
        time_random_reset: Optional[bool] = False,
        time_reset_prob: Optional[float] = 0.01,
    ):
        self.task = task
        self.model_name = model_name
        self.eval_batch_size = eval_batch_size
        self.result_path = result_path
        self.lr = lr
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.reg_lambda = reg_lambda
        self.time_random_reset = time_random_reset
        self.time_reset_prob = time_reset_prob


def continual_test_time_adaptation_tent(args, model, tokenizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    batch_size = args.eval_batch_size

    lm_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device_map="auto",
    )

    trainable_initial = _collect_trainable_initial_params(model)
    if len(trainable_initial) == 0:
        raise ValueError("No trainable parameters found. Did you inject LoRA?")

    if isinstance(args.task, str):
        task_list = [t.strip() for t in args.task.split(",") if t.strip()]
    else:
        task_list = args.task

    all_task_results = {}

    for task_name in task_list:
        print(f"\n[TTA-TENT] Start task: {task_name}")
        task_dict = get_task_dict([task_name])
        task = task_dict[task_name]
        docs = list(task.validation_docs())
        n_examples = len(docs)

        pre_acc_list = []
        memory_list = []
        time_list = []

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )

        for i in range(0, n_examples, batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, n_examples)))
            samples_dict = {task_name: batch_indices}

            with torch.no_grad():
                pre_results = simple_evaluate(
                    model=lm_model,
                    tasks=[task_name],
                    samples=samples_dict,
                    batch_size=len(batch_docs),
                    num_fewshot=0,
                    random_seed=0,
                    torch_random_seed=0,
                )
            pre_task_result = pre_results.get("results", {}).get(task_name)
            if pre_task_result is None:
                print(f"[Warning] task {task_name} not in pre_results, skipping batch")
                continue

            pre_acc, _ = get_main_metric(task, pre_task_result)
            pre_acc_list.append(pre_acc)

            model.train()

            requests_list = []
            task_family = get_task_family(task)

            for doc in batch_docs:
                ctx = task.fewshot_context(doc, num_fewshot=0)
                requests = task.construct_requests(doc, ctx)
                requests_list.append(requests)

            batch_contexts = construct_context(
                requests_list=requests_list,
                tokenizer=tokenizer,
                model=model,
                task_family=task_family,
                device=device,
            )

            encoded = tokenizer(
                batch_contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            start_time = time.time()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
            loss_entropy = torch.sum(entropy * attention_mask.float()) / (
                torch.sum(attention_mask) + 1e-10
            )

            reg_loss = torch.tensor(0.0, device=device)
            if args.reg_lambda > 0 or args.time_random_reset:
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    init_val = trainable_initial.get(name)
                    if init_val is None:
                        continue
                    if args.reg_lambda > 0:
                        reg_loss = reg_loss + torch.norm(p - init_val.to(p.device)) ** 2
                    if args.time_random_reset and args.time_reset_prob > 0:
                        mask = torch.rand(p.shape, device=p.device) < args.time_reset_prob
                        if mask.any():
                            p.data = torch.where(mask, init_val.to(p.device), p.data)

            loss = loss_entropy + args.reg_lambda * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                memory_list.append(torch.cuda.memory_allocated() / 1024 ** 3)
            time_list.append((time.time() - start_time) / max(input_ids.size(0), 1))

            del outputs, probs, entropy, loss, loss_entropy, reg_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.eval()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        summary = {
            "task": task_name,
            "pre_acc_mean": sum(pre_acc_list) / len(pre_acc_list) if pre_acc_list else 0.0,
            "cuda_memory_gb": sum(memory_list) / len(memory_list) if memory_list else 0.0,
            "time_per_sample": sum(time_list) / len(time_list) if time_list else 0.0,
        }
        print(f"[TTA-TENT] Task {task_name} summary: {summary}")
        all_task_results[task_name] = summary

    return all_task_results


def run_tent(
    task: str,
    model_name: Optional[str] = "./Qwen1.5-MoE-A2.7B-Chat",
    eval_batch_size: Optional[int] = 4,
    result_path: Optional[str] = None,
    lr: Optional[float] = 1e-5,
    lora_r: Optional[int] = 16,
    lora_alpha: Optional[int] = 16,
    reg_lambda: Optional[float] = 0.0,
    time_random_reset: Optional[bool] = False,
    time_reset_prob: Optional[float] = 0.01,
):
    torch.manual_seed(0)

    args = Args(
        task=task,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        result_path=result_path,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        reg_lambda=reg_lambda,
        time_random_reset=time_random_reset,
        time_reset_prob=time_reset_prob,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = inject_lora_into_moe(model, r=args.lora_r, alpha=args.lora_alpha)
    freeze_non_lora_params(model)

    start_time = time.time()
    results = continual_test_time_adaptation_tent(args, model, tokenizer)
    total_time = time.time() - start_time

    print(
        f"[TTA-TENT] Results: {results} | Time: {total_time:.2f} seconds | "
        f"Number of parameters: {model.num_parameters()}"
    )

    if result_path is not None:
        result_dir = os.path.dirname(result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    Fire(run_tent)
