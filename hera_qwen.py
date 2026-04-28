# -*- coding: utf-8 -*-
import json
import os
import time
from typing import Optional

import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer

from module.HERA_CTTA_MoE import continual_test_time_adaptation_hera
from module.moe_lora import freeze_non_lora_params, inject_lora_into_moe


class Args:
    def __init__(
        self,
        task,
        model_name: Optional[str] = "./Qwen1.5-MoE-A2.7B-Chat",
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 4,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        num_fewshot: Optional[int] = 0,
        lr: Optional[float] = 1e-5,
        target_layer: Optional[int] = 0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 16,
        warmup_batches: Optional[int] = 10,
        tau_low: Optional[float] = 0.35,
        tau_high: Optional[float] = 0.65,
        kappa: Optional[float] = 0.3,
        safe_top_b: Optional[int] = 4,
        tau_d: Optional[float] = 0.2,
        tau_i: Optional[float] = 0.2,
        alpha_d: Optional[float] = 0.5,
        alpha_i: Optional[float] = 1.0,
        alpha_h: Optional[float] = 0.5,
        lambda_q: Optional[float] = 1.0,
        lambda_c: Optional[float] = 1.0,
        lambda_router: Optional[float] = 2.0,
        lambda_rho: Optional[float] = 1.0,
        lambda_h: Optional[float] = 1.0,
        lambda_bal: Optional[float] = 0.05,
        lambda_anchor: Optional[float] = 1e-4,
        restore_rho: Optional[float] = 0.1,
        a_n: Optional[float] = 0.5,
        a_q: Optional[float] = 0.25,
        a_c: Optional[float] = 0.25,
        m_min: Optional[float] = 0.1,
        m_max: Optional[float] = 1.0,
    ):
        self.task = task
        self.model_name = model_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.output_path = output_path
        self.result_path = result_path
        self.model_path = model_path
        self.num_fewshot = num_fewshot
        self.lr = lr
        self.target_layer = target_layer
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        self.warmup_batches = warmup_batches
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.kappa = kappa
        self.safe_top_b = safe_top_b
        self.tau_d = tau_d
        self.tau_i = tau_i
        self.alpha_d = alpha_d
        self.alpha_i = alpha_i
        self.alpha_h = alpha_h
        self.lambda_q = lambda_q
        self.lambda_c = lambda_c
        self.lambda_router = lambda_router
        self.lambda_rho = lambda_rho
        self.lambda_h = lambda_h
        self.lambda_bal = lambda_bal
        self.lambda_anchor = lambda_anchor
        self.restore_rho = restore_rho
        self.a_n = a_n
        self.a_q = a_q
        self.a_c = a_c
        self.m_min = m_min
        self.m_max = m_max


def run_hera(
    task: str,
    model_name: Optional[str] = "./Qwen1.5-MoE-A2.7B-Chat",
    train_batch_size: Optional[int] = 4,
    eval_batch_size: Optional[int] = 4,
    output_path: Optional[str] = None,
    result_path: Optional[str] = None,
    model_path: Optional[str] = None,
    num_fewshot: Optional[int] = 0,
    lr: Optional[float] = 1e-5,
    target_layer: Optional[int] = 0,
    lora_r: Optional[int] = 16,
    lora_alpha: Optional[int] = 16,
    warmup_batches: Optional[int] = 10,
    tau_low: Optional[float] = 0.35,
    tau_high: Optional[float] = 0.65,
    kappa: Optional[float] = 0.3,
    safe_top_b: Optional[int] = 4,
    tau_d: Optional[float] = 0.2,
    tau_i: Optional[float] = 0.2,
    alpha_d: Optional[float] = 0.5,
    alpha_i: Optional[float] = 1.0,
    alpha_h: Optional[float] = 0.5,
    lambda_q: Optional[float] = 1.0,
    lambda_c: Optional[float] = 1.0,
    lambda_router: Optional[float] = 2.0,
    lambda_rho: Optional[float] = 1.0,
    lambda_h: Optional[float] = 1.0,
    lambda_bal: Optional[float] = 0.05,
    lambda_anchor: Optional[float] = 1e-4,
    restore_rho: Optional[float] = 0.1,
    a_n: Optional[float] = 0.5,
    a_q: Optional[float] = 0.25,
    a_c: Optional[float] = 0.25,
    m_min: Optional[float] = 0.1,
    m_max: Optional[float] = 1.0,
):
    torch.manual_seed(0)

    args = Args(
        task=task,
        model_name=model_name,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        output_path=output_path,
        result_path=result_path,
        model_path=model_path,
        num_fewshot=num_fewshot,
        lr=lr,
        target_layer=target_layer,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        warmup_batches=warmup_batches,
        tau_low=tau_low,
        tau_high=tau_high,
        kappa=kappa,
        safe_top_b=safe_top_b,
        tau_d=tau_d,
        tau_i=tau_i,
        alpha_d=alpha_d,
        alpha_i=alpha_i,
        alpha_h=alpha_h,
        lambda_q=lambda_q,
        lambda_c=lambda_c,
        lambda_router=lambda_router,
        lambda_rho=lambda_rho,
        lambda_h=lambda_h,
        lambda_bal=lambda_bal,
        lambda_anchor=lambda_anchor,
        restore_rho=restore_rho,
        a_n=a_n,
        a_q=a_q,
        a_c=a_c,
        m_min=m_min,
        m_max=m_max,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = inject_lora_into_moe(model, r=args.lora_r, alpha=args.lora_alpha)
    freeze_non_lora_params(model)

    start_time = time.time()
    results = continual_test_time_adaptation_hera(
        args=args,
        model=model,
        tokenizer=tokenizer,
        layers_to_use=[args.target_layer],
    )
    total_time = time.time() - start_time

    print(
        f"[HERA] Results: {results} | Time: {total_time:.2f} seconds | "
        f"Number of parameters: {model.num_parameters()}"
    )

    if result_path is not None:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    Fire(run_hera)
