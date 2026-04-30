# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
from datetime import datetime
from typing import Optional

import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict

from module.Construct_context import construct_context
from module.moe_lora import inject_lora_into_moe, freeze_non_lora_params


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
        value = task_result.get("exact_match,none", 0.0)
        return value, value
    raise ValueError(f"Unknown OUTPUT_TYPE: {task.OUTPUT_TYPE}")


def infer_default_eval_batch_size(model_name: str) -> int:
    model_name_lower = model_name.lower()
    if "deepseek" in model_name_lower:
        return 4
    return 8


def collect_trainable_initial_params(model):
    return {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


class Args:
    def __init__(
        self,
        task,
        model_name: Optional[str] = "./Qwen1.5-MoE-A2.7B-Chat",
        eval_batch_size: Optional[int] = None,
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
        self.eval_batch_size = (
            infer_default_eval_batch_size(model_name)
            if eval_batch_size is None
            else eval_batch_size
        )
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

    trainable_initial = collect_trainable_initial_params(model)
    if len(trainable_initial) == 0:
        raise ValueError(
            "No trainable parameters found. Tent requires trainable LoRA parameters. "
            "Please check the layer-selection diagnostics printed by module/moe_lora.py."
        )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )

    if isinstance(args.task, str):
        task_list = [task.strip() for task in args.task.split(",") if task.strip()]
    else:
        task_list = args.task

    all_task_results = {}

    for task_name in task_list:
        print(f"\n[TENT] Start task: {task_name}")
        task_dict = get_task_dict([task_name])
        task = task_dict[task_name]
        docs = list(task.validation_docs())
        n_examples = len(docs)

        pre_acc_list = []
        memory_list = []
        time_list = []

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
                print(f"[TENT][Warning] task {task_name} not in pre_results, skipping batch")
                continue

            pre_acc, _ = get_main_metric(task, pre_task_result)
            pre_acc_list.append(pre_acc)

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
            model.train()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            probs = torch.softmax(outputs.logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
            loss_entropy = torch.sum(entropy * attention_mask.float()) / (
                torch.sum(attention_mask) + 1e-10
            )

            reg_loss = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in trainable_initial:
                    continue
                init_val = trainable_initial[name].to(param.device)
                if args.reg_lambda > 0:
                    reg_loss = reg_loss + torch.norm(param - init_val) ** 2
                if args.time_random_reset and args.time_reset_prob > 0:
                    mask = torch.rand_like(param) < args.time_reset_prob
                    param.data = torch.where(mask, init_val, param.data)

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
        print(f"[TENT] Task {task_name} summary: {summary}")
        all_task_results[task_name] = summary

    return all_task_results


def run_tent(
    task: str,
    model_name: Optional[str] = "./Qwen1.5-MoE-A2.7B-Chat",
    eval_batch_size: Optional[int] = None,
    result_path: Optional[str] = "results/tent_results.txt",
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

    start_dt = datetime.now()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if hasattr(model, "config") and model.config is not None:
        model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.use_cache = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    model = inject_lora_into_moe(model, r=args.lora_r, alpha=args.lora_alpha)
    freeze_non_lora_params(model)

    print("========== Starting Tent Evaluation ==========")
    print(f"Start Time:   {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Path:   {args.model_name}")
    print(f"Task:         {task}")
    print(f"Batch Size:   {args.eval_batch_size}")
    print(f"LR:           {args.lr}")
    print(f"LoRA (r/a):   {args.lora_r}/{args.lora_alpha}")
    print(f"Reg Lambda:   {args.reg_lambda}")
    print(f"Rand Reset:   {args.time_random_reset}")
    print(f"Reset Prob:   {args.time_reset_prob}")
    print("=============================================")

    start_ts = time.time()
    results = continual_test_time_adaptation_tent(args, model, tokenizer)
    total_seconds = time.time() - start_ts
    end_dt = datetime.now()

    if torch.cuda.is_available():
        peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    else:
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0

    print(
        f"[TENT] Results: {results} | Time: {total_seconds:.2f} seconds | "
        f"Peak Alloc: {peak_allocated_gb:.2f} GB | Peak Reserved: {peak_reserved_gb:.2f} GB"
    )

    if args.result_path is not None:
        result_dir = os.path.dirname(args.result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        with open(args.result_path, "w", encoding="utf-8") as f:
            f.write("Tent Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Time (s): {total_seconds:.2f}\n")
            f.write(f"Peak GPU Memory Allocated (GB): {peak_allocated_gb:.2f}\n")
            f.write(f"Peak GPU Memory Reserved (GB):  {peak_reserved_gb:.2f}\n")
            f.write(f"Model Path: {args.model_name}\n")
            f.write(f"Tasks: {task}\n")
            f.write(f"Eval Batch Size: {args.eval_batch_size}\n")
            f.write(f"LR: {args.lr}\n")
            f.write(f"LoRA r: {args.lora_r}\n")
            f.write(f"LoRA alpha: {args.lora_alpha}\n")
            f.write(f"Reg Lambda: {args.reg_lambda}\n")
            f.write(f"Random Reset: {args.time_random_reset}\n")
            f.write(f"Reset Prob: {args.time_reset_prob}\n")
            f.write("=" * 50 + "\n")
            for task_name, summary in results.items():
                f.write(f"{task_name}: {summary}\n")


if __name__ == "__main__":
    Fire(run_tent)
