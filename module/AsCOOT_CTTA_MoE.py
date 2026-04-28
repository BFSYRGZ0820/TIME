import json
import os
import gc
import sys
import time
import pickle
from typing import Optional
import copy
import logging
import torch
import math
from itertools import islice
import numpy as np
from sympy import linear_eq_to_matrix
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lm_eval.tasks import get_task_dict
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from module.As_COOT import AsCOOT
from module.Expert_encoder_cache import ExpertEncodingCache
from module.Construct_context import construct_context

def get_task_family(task):

    output_type = getattr(task, "OUTPUT_TYPE", None)
    if output_type in ["loglikelihood", "multiple_choice"]:
        return "discriminative"
    elif output_type in ["generate_until"]:
        return "generative"
    else:
        raise ValueError(f"Unknown task OUTPUT_TYPE: {output_type}")


def get_main_metric(task, task_result):

    if task.OUTPUT_TYPE in ["loglikelihood", "multiple_choice"]:
        return task_result.get("acc,none", 0.0), task_result.get("acc_stderr,none", 0.0)
    elif task.OUTPUT_TYPE == "generate_until":
        return task_result.get("exact_match,none", 0.0), task_result.get("exact_match,none", 0.0)
    else:
        raise ValueError(f"Unknown OUTPUT_TYPE: {task.OUTPUT_TYPE}")


def continual_test_time_adaptation(args, model, tokenizer, routed_experts,
                                   sample_encoder=None, expert_encoder=None,
                                   layers_to_use=[0], device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    batch_size = args.eval_batch_size
    para_initial = {}
    all_task_results = {}

    lm_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device_map="auto"
    )

    # As-COOT
    solver = AsCOOT(epsilon=args.epsilon, rho_x=0.1, max_iter_outer=50, max_iter_inner=200)

    # prepare expert cache if encoders provided
    expert_cache = None
    use_ascoot = (sample_encoder is not None and expert_encoder is not None)
    if use_ascoot:
        try:
            sample_encoder.to('cpu')
            expert_encoder.to('cpu')
        except Exception:
            pass
        sample_encoder.eval()
        expert_encoder.eval()
        expert_cache = ExpertEncodingCache(expert_encoder)

    ### parameters initialization
    for name, p in model.named_parameters():
        if ".layers." not in name:
            continue
        try:
            # 动态解析层数，无视 peft 加了多少前缀
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
        except ValueError:
            continue
            
        if layer_idx not in layers_to_use:
            continue
            
        # 收集 Router (gate) 的参数
        if ".mlp.gate." in name:
            para_initial[name] = p.detach().clone().cpu()
        # 收集专家 (experts) 的 LoRA 参数
        if ".mlp.experts." in name and "lora" in name.lower():
            para_initial[name] = p.detach().clone().cpu()

    if isinstance(args.task, str):
        task_list = args.task.split(",")
    else:
        task_list = args.task

    for task_name in task_list:
        print(f"\n[TTA] Start task: {task_name}")
        task_dict = get_task_dict([task_name])
        task = task_dict[task_name]
        docs = list(task.validation_docs())
        n_examples = len(docs)
        pre_acc_list, post_acc_list = [], []
        memory_list = []
        time_list = []

        selected_total = 0
        step_counter = 0
        for i in range(0, n_examples, batch_size):
            optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=1e-5)
            batch_docs = docs[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, n_examples)))
            samples_dict = {task_name: batch_indices}

            # ---- 1. Pre-eval (eval 模式, no grad) ----
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
            pre_task_result = pre_results.get('results', {}).get(task_name)
            if pre_task_result is None:
                print(f"[Warning] task {task_name} not in pre_results, skipping batch")
                continue

            pre_acc, pre_stderr = get_main_metric(task, pre_task_result)
            pre_acc_list.append(pre_acc)
            # ---- 1. TTA: entropy minimization ----
            model.train()

            # ---- construct input ----
            requests_list = []
            task_family = get_task_family(task)

            def get_ctx_for_doc(task, doc, num_fewshot=0):
                return task.fewshot_context(doc, num_fewshot=0)

            for doc in batch_docs:
                ctx = get_ctx_for_doc(task, doc)
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
                truncation=True
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            selected_indices = torch.arange(input_ids.size(0))

            selected_total += len(selected_indices)
            start_time = time.time()
            # ---- entropy minimization steps----
            if len(selected_indices) > 0:
                sel_input_ids = input_ids[selected_indices].to(device)
                sel_attention_mask = attention_mask[selected_indices].to(device)
                sel_texts = [batch_contexts[idx] for idx in selected_indices.tolist()]

                ### As-COOT gian pi_s
                if use_ascoot:
                    with torch.no_grad():
                        X_cpu = sample_encoder(sel_texts)
                        if isinstance(X_cpu, torch.Tensor):
                            X_cpu = X_cpu.detach().cpu().float().clone()
                        else:
                            X_cpu = torch.tensor(X_cpu, dtype=torch.float32)
                    X = X_cpu.to(device)
                else:
                    X = None
                layer_to_pi_s = {}

                for layer_idx in layers_to_use:
                    if not use_ascoot:
                        continue
                    # if step_counter % 3 == 0:
                    keys, Y_cpu = expert_cache.ensure_layer_encoded(model, layer_idx)
                    Y = Y_cpu.to(device)
                    try:
                        with torch.no_grad():
                            pi_s, _ = solver(X, Y)
                            #print(f"[监控] pi_s MAX: {pi_s.max().item():.5f}, MIN: {pi_s.min().item():.5f}")
                    finally:
                        del Y
                        torch.cuda.empty_cache()

                    layer_to_pi_s[layer_idx] = pi_s  # (B, E)

                ### pi_s-based outliers filtering

                PI_OUTLIER_RATIO = 0.01
                B_before = sel_input_ids.size(0)

                if len(layer_to_pi_s) > 0:
                    combined_keep_mask = torch.ones(B_before, dtype=torch.bool)
                    for layer_idx, pi_s in layer_to_pi_s.items():
                        try:
                            row_sums = pi_s.sum(dim=1).detach().cpu()
                        except Exception:
                            row_sums = pi_s.detach().cpu().sum(dim=1)
                        mean_rs = row_sums.mean().item()
                        thr = mean_rs * PI_OUTLIER_RATIO
                        keep_mask_layer = (row_sums >= thr)
                        if keep_mask_layer.sum().item() == 0:
                            keep_mask_layer = torch.ones_like(keep_mask_layer)
                        combined_keep_mask = combined_keep_mask & keep_mask_layer

                    if combined_keep_mask.sum().item() == 0:
                        combined_keep_mask = torch.ones(B_before, dtype=torch.bool)

                    keep_indices_local = combined_keep_mask.nonzero(as_tuple=True)[0].tolist()

                    if len(keep_indices_local) < B_before:
                        sel_input_ids = sel_input_ids[keep_indices_local].to(device)
                        sel_attention_mask = sel_attention_mask[keep_indices_local].to(device)
                        sel_texts = [sel_texts[i] for i in keep_indices_local]
                        for layer_idx in list(layer_to_pi_s.keys()):
                            pi_s = layer_to_pi_s[layer_idx]
                            pi_s_cpu = pi_s.detach().cpu()
                            pi_s_cpu = pi_s_cpu[keep_indices_local, :]
                            layer_to_pi_s[layer_idx] = pi_s_cpu.to(device)
                    else:
                        for layer_idx in list(layer_to_pi_s.keys()):
                            layer_to_pi_s[layer_idx] = layer_to_pi_s[layer_idx].detach().to(device)

                    B_after = sel_input_ids.size(0)
                else:  # No pi_s (use_ascoot=False),No filtering
                    sel_input_ids = sel_input_ids.to(device)
                    sel_attention_mask = sel_attention_mask.to(device)
                    B_after = sel_input_ids.size(0)

                if B_after == 0:
                    print("[Info] All selected samples filtered as outliers; skipping this update step.")
                    model.eval()
                    continue

                ## As-COOT hook (Modify the routing logits)
                hooks = []
                train_gate_logits = {}

                def make_ucoot_hook(layer_idx, pi_s, ot_strength=5.0):

                    def hook(module, inputs, outputs):
                        B = sel_input_ids.size(0)
                        E = outputs.size(-1)
                        seq_len = outputs.view(B, -1, E).size(1)
                        logits_orig = outputs.view(B, seq_len, E)
                        train_gate_logits[layer_idx] = logits_orig
                        pi_dist = pi_s.to(logits_orig.device)
                        pi_dist = pi_dist / (pi_dist.sum(dim=-1, keepdim=True) + 1e-12)
                        ot_bias = torch.log(pi_dist + 1e-12)  # (B, E)
                        ot_bias = ot_bias.unsqueeze(1)
                        logits_new = logits_orig + ot_strength * ot_bias
                        return logits_new.view(-1, E)

                    return hook

                train_gate_logits.clear()

                for layer_idx in layers_to_use:
                    pi_s = layer_to_pi_s.get(layer_idx, None)
                    if pi_s is None:
                        continue
                    h = model.base_model.model.model.layers[layer_idx].mlp.gate.register_forward_hook(
                        make_ucoot_hook(layer_idx, pi_s, args.ot_strength)
                    )
                    hooks.append(h)

                outputs = model(input_ids=sel_input_ids, attention_mask=sel_attention_mask)

                # remove hooks
                for h in hooks:
                    h.remove()

                trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
                if len(trainable_params) == 0:
                    print("trainable params == 0")
                    model.eval()
                else:
                    optimizer.param_groups[0]["params"] = trainable_params

                    ### entropy loss
                    probs = torch.softmax(outputs.logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                    loss_entropy = torch.sum(entropy * sel_attention_mask.float()) / (
                            torch.sum(sel_attention_mask) + 1e-10)

                    ### reg loss
                    reg_loss = torch.tensor(0.0, device=device)
                    for name, p in model.named_parameters():
                        if p.requires_grad and name in para_initial:
                            reg_loss = reg_loss + torch.norm(p - para_initial[name].to(p.device)) ** 2
                            mask = (torch.rand_like(p) < 0.01).float()
                            init_val = para_initial[name].to(device)
                            p.data = torch.where(mask == 1, init_val, p.data)

                    l2_lambda = 0.1
                    loss = loss_entropy + l2_lambda * reg_loss

                    memory_list.append((torch.cuda.memory_allocated())/1024**3)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                del outputs, probs, entropy, loss, loss_entropy, reg_loss
                torch.cuda.empty_cache()
                model.eval()

            time_list.append((time.time()-start_time) / batch_size)
            step_counter += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        # ---- 3. Summary ----
        summary = {
            "task": task_name,
            "pre_acc_mean": sum(pre_acc_list) / len(pre_acc_list) if pre_acc_list else 0.0,
            "cuda memory (G)": sum(memory_list) / len(memory_list) if memory_list else 0.0,
            "time / sample": sum(time_list) / len(time_list) if time_list else 0.0,
        }
        print(f"[TTA] Task {task_name} summary: {summary}")
        all_task_results[task_name] = summary

    return all_task_results



