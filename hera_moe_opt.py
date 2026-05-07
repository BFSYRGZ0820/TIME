# -*- coding: utf-8 -*-
import os
import re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from fire import Fire
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from module.Construct_context import construct_context
from module.moe_lora import freeze_non_lora_params, inject_lora_into_moe


_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


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
    return 4 if "deepseek" in model_name.lower() else 8


def collect_trainable_initial_params(model):
    return {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def cosine_prior(query: torch.Tensor, keys: torch.Tensor, temperature: float) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    keys = F.normalize(keys, dim=-1)
    logits = query @ keys.T
    return torch.softmax(logits / temperature, dim=-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * mask).sum(dim=dim) / denom.squeeze(dim)


class RunningZScore:
    def __init__(self, momentum: float = 0.95, eps: float = 1e-6):
        self.momentum = momentum
        self.eps = eps
        self.mean = None
        self.var = None

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        detached = values.detach()
        cur_mean = detached.mean()
        cur_var = detached.var(unbiased=False)
        if self.mean is None:
            self.mean = cur_mean
            self.var = cur_var
        else:
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * cur_mean
            self.var = self.momentum * self.var + (1.0 - self.momentum) * cur_var
        return (values - self.mean) / torch.sqrt(self.var + self.eps)


class RunningStatsBank:
    def __init__(self):
        self.stats = {
            "pred_entropy": RunningZScore(),
            "route_entropy": RunningZScore(),
            "disagreement": RunningZScore(),
            "proto_dist": RunningZScore(),
        }

    def normalize(self, key: str, values: torch.Tensor) -> torch.Tensor:
        return self.stats[key].normalize(values)


class DomainStateTracker:
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.prev_mu = None
        self.prev_rho = None
        self.prev_pred_entropy = None
        self.state = None
        self.last_delta = 0.0

    def update(self, mu_t: torch.Tensor, rho_t: torch.Tensor, mean_pred_entropy: float,
               lambda_rho: float = 1.0, lambda_h: float = 1.0) -> float:
        if self.prev_mu is None:
            delta_t = torch.tensor(0.0, device=mu_t.device)
        else:
            delta_t = (
                torch.norm(mu_t - self.prev_mu, p=2)
                + lambda_rho * torch.norm(rho_t - self.prev_rho, p=1)
                + lambda_h * torch.abs(torch.tensor(mean_pred_entropy, device=mu_t.device) - self.prev_pred_entropy)
            )
        state_vec = torch.cat([
            mu_t.detach(),
            rho_t.detach(),
            torch.tensor([mean_pred_entropy, delta_t.item()], device=mu_t.device),
        ], dim=0)
        if self.state is None:
            self.state = state_vec
        else:
            self.state = self.momentum * self.state + (1.0 - self.momentum) * state_vec
        self.prev_mu = mu_t.detach()
        self.prev_rho = rho_t.detach()
        self.prev_pred_entropy = torch.tensor(mean_pred_entropy, device=mu_t.device)
        self.last_delta = float(delta_t.item())
        return self.last_delta


class ExpertStateBank:
    def __init__(self, num_experts: int, hidden_dim: int, device: torch.device):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.device = device
        self.anchor = torch.zeros(num_experts, hidden_dim, device=device)
        self.proto = torch.zeros(num_experts, hidden_dim, device=device)
        self.reliability = torch.full((num_experts,), 0.5, device=device)
        self.contamination = torch.zeros(num_experts, device=device)
        self.initialized = torch.zeros(num_experts, dtype=torch.bool, device=device)

    def ensure_initialized(self, fallback: torch.Tensor):
        fallback = fallback.detach().to(self.device)
        missing = ~self.initialized
        if missing.any():
            self.anchor[missing] = fallback
            self.proto[missing] = fallback
            self.initialized[missing] = True

    def warmup_update(self, sample_repr: torch.Tensor, route_probs: torch.Tensor, safe_mask: torch.Tensor):
        top1 = route_probs.argmax(dim=-1)
        for expert_idx in range(self.num_experts):
            mask = (top1 == expert_idx) & safe_mask
            if mask.any():
                mean_repr = sample_repr[mask].mean(dim=0)
                if not self.initialized[expert_idx]:
                    self.anchor[expert_idx] = mean_repr
                    self.proto[expert_idx] = mean_repr
                    self.initialized[expert_idx] = True
                else:
                    self.anchor[expert_idx] = 0.9 * self.anchor[expert_idx] + 0.1 * mean_repr
                    self.proto[expert_idx] = 0.9 * self.proto[expert_idx] + 0.1 * mean_repr

    def blended_proto(self, delta_t: float, delta_scale: float = 1.0,
                      gamma_min: float = 0.2, gamma_max: float = 0.9) -> torch.Tensor:
        gamma = delta_t / (delta_t + delta_scale) if delta_t > 0 else 0.0
        gamma = float(max(gamma_min, min(gamma_max, gamma)))
        return gamma * self.anchor + (1.0 - gamma) * self.proto

    def health_prior(self, lambda_q: float, lambda_c: float) -> torch.Tensor:
        logits = lambda_q * self.reliability - lambda_c * self.contamination
        return torch.softmax(logits, dim=0)

    def safe_experts(self, top_b: int) -> torch.Tensor:
        scores = self.reliability - self.contamination
        safe_k = min(top_b, scores.numel())
        indices = torch.topk(scores, k=safe_k, dim=0).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[indices] = True
        return mask

    def update_after_step(self, sample_repr: torch.Tensor, sample_prior: torch.Tensor, risk: torch.Tensor,
                          low_threshold: float, high_threshold: float,
                          eta_e: float, eta_q: float, eta_c: float):
        top2 = torch.topk(sample_prior, k=min(2, self.num_experts), dim=-1).indices
        for expert_idx in range(self.num_experts):
            assigned = (top2 == expert_idx).any(dim=-1)
            safe_mask = assigned & (risk < low_threshold)
            risky_mask = assigned & (risk >= high_threshold)

            if safe_mask.any():
                safe_repr = sample_repr[safe_mask].mean(dim=0)
                self.proto[expert_idx] = eta_e * self.proto[expert_idx] + (1.0 - eta_e) * safe_repr
                safe_risk = risk[safe_mask].mean()
                self.reliability[expert_idx] = eta_q * self.reliability[expert_idx] + (1.0 - eta_q) * (1.0 - safe_risk)

            denom = safe_mask.sum() + risky_mask.sum()
            if denom.item() > 0:
                contam = risky_mask.sum().float() / denom.float()
                self.contamination[expert_idx] = eta_c * self.contamination[expert_idx] + (1.0 - eta_c) * contam


def find_target_gate_module(model, layer_idx: int):
    candidate_paths = [
        f"model.layers.{layer_idx}.mlp.gate",
        f"base_model.model.model.layers.{layer_idx}.mlp.gate",
        f"base_model.model.layers.{layer_idx}.mlp.gate",
    ]
    modules = dict(model.named_modules())
    for path in candidate_paths:
        if path in modules:
            return modules[path], path
    for name, module in modules.items():
        if name.endswith(f"layers.{layer_idx}.mlp.gate"):
            return module, name
    raise ValueError(f"Cannot find gate module for layer {layer_idx}")


def register_probe_hooks(model, layer_idx: int, storage: Dict[str, torch.Tensor]):
    gate_module, _ = find_target_gate_module(model, layer_idx)

    def pre_hook(module, inputs):
        storage["pre_router_hidden"] = inputs[0].detach()

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            raise RuntimeError(
                "Current HERA probe hook expects gate logits tensor, but got tuple outputs. "
                "This model's gate forward needs a model-specific patch."
            )
        storage["router_logits"] = outputs.detach()

    handles = [
        gate_module.register_forward_pre_hook(pre_hook),
        gate_module.register_forward_hook(hook),
    ]
    return handles


def register_calibration_hook(model, layer_idx: int, bias: torch.Tensor):
    gate_module, _ = find_target_gate_module(model, layer_idx)

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            raise RuntimeError(
                "HERA calibration hook only supports tensor gate logits. "
                "Got tuple outputs; please patch this model's gate forward."
            )

        b = bias
        if b.dim() == 2:
            b = b.unsqueeze(1)

        bsz = b.size(0)
        num_experts = b.size(-1)

        if outputs.dim() == 2:
            total_tokens = outputs.size(0)
            if total_tokens % bsz != 0:
                raise RuntimeError(
                    f"Gate output shape {tuple(outputs.shape)} cannot match bias {tuple(b.shape)}"
                )
            seq_len = total_tokens // bsz
            logits = outputs.view(bsz, seq_len, num_experts)
            b = b.to(device=outputs.device, dtype=outputs.dtype)
            b = b.expand(bsz, seq_len, num_experts)
            return (logits + b).reshape(-1, num_experts)

        if outputs.dim() == 3:
            b = b.to(device=outputs.device, dtype=outputs.dtype)
            return outputs + b.expand_as(outputs)

        raise RuntimeError(f"Unsupported gate output dim: {outputs.dim()}")

    return gate_module.register_forward_hook(hook)


def sample_mean_hidden(pre_router_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    hidden = pre_router_hidden.view(batch_size, seq_len, -1)
    mask = attention_mask.unsqueeze(-1)
    return masked_mean(hidden, mask, dim=1)


def mean_route_probs(router_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    num_experts = router_logits.size(-1)
    logits = router_logits.view(batch_size, seq_len, num_experts)
    probs = torch.softmax(logits, dim=-1)
    mask = attention_mask.unsqueeze(-1)
    return masked_mean(probs, mask, dim=1)


def per_sample_prediction_entropy(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    mask = attention_mask.float()
    return (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def route_entropy(route_probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(route_probs * torch.log(route_probs + 1e-12), dim=-1)


def expert_disagreement(route_probs: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(route_probs, k=min(2, route_probs.size(-1)), dim=-1).values
    if top2.size(-1) < 2:
        return torch.zeros(route_probs.size(0), device=route_probs.device)
    return 1.0 - (top2[:, 0] - top2[:, 1]).clamp_min(0.0)


def prototype_mismatch(sample_repr: torch.Tensor, expert_repr: torch.Tensor) -> torch.Tensor:
    sample_repr = F.normalize(sample_repr.float(), dim=-1)
    expert_repr = F.normalize(expert_repr.float(), dim=-1)
    sim = sample_repr @ expert_repr.T
    max_sim = sim.max(dim=-1).values
    return 1.0 - max_sim


def compute_risk(pred_entropy: torch.Tensor, route_entropy_val: torch.Tensor,
                 disagreement: torch.Tensor, proto_dist: torch.Tensor,
                 running_stats: RunningStatsBank) -> torch.Tensor:
    z1 = running_stats.normalize("pred_entropy", pred_entropy)
    z2 = running_stats.normalize("route_entropy", route_entropy_val)
    z3 = running_stats.normalize("disagreement", disagreement)
    z4 = running_stats.normalize("proto_dist", proto_dist)
    return torch.sigmoid(1.0 * z1 + 0.5 * z2 + 0.75 * z3 + 1.0 * z4)


def sample_update_weights(risk: torch.Tensor, tau_low: float, tau_high: float, kappa: float) -> torch.Tensor:
    ones = torch.ones_like(risk)
    mid = torch.full_like(risk, kappa)
    zeros = torch.zeros_like(risk)
    return torch.where(risk < tau_low, ones, torch.where(risk < tau_high, mid, zeros))


def build_sample_prior(sample_repr: torch.Tensor, batch_repr: torch.Tensor, expert_repr: torch.Tensor,
                       state_bank: ExpertStateBank, risk: torch.Tensor,
                       tau_d: float, tau_i: float,
                       alpha_d: float, alpha_i: float, alpha_h: float,
                       lambda_q: float, lambda_c: float, safe_top_b: int, tau_high: float) -> torch.Tensor:
    domain_prior = cosine_prior(batch_repr.unsqueeze(0), expert_repr, tau_d).squeeze(0)
    instance_prior = cosine_prior(sample_repr, expert_repr, tau_i)
    health_prior = state_bank.health_prior(lambda_q=lambda_q, lambda_c=lambda_c)

    sample_prior = (
        domain_prior.unsqueeze(0).pow(alpha_d)
        * instance_prior.pow(alpha_i)
        * health_prior.unsqueeze(0).pow(alpha_h)
    )
    sample_prior = sample_prior / sample_prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    safe_mask = state_bank.safe_experts(top_b=safe_top_b)
    high_risk = risk >= tau_high
    if high_risk.any():
        sample_prior[high_risk] = sample_prior[high_risk] * safe_mask.unsqueeze(0).float()
        sample_prior[high_risk] = sample_prior[high_risk] / sample_prior[high_risk].sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return sample_prior


def entropy_loss_per_sample(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    mask = attention_mask.float()
    return (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def balance_regularization(sample_prior: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    total_weight = weights.sum().clamp_min(1e-12)
    avg_prior = (weights.unsqueeze(-1) * sample_prior).sum(dim=0) / total_weight
    uniform = torch.full_like(avg_prior, 1.0 / avg_prior.numel())
    return F.kl_div(torch.log(avg_prior.clamp_min(1e-12)), uniform, reduction="sum")


def _extract_expert_index(name: str) -> Optional[int]:
    match = _EXPERT_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def anchor_regularization(model, initial_params: Dict[str, torch.Tensor],
                          state_bank: ExpertStateBank) -> torch.Tensor:
    reg = None
    weights = (state_bank.reliability + state_bank.contamination).detach()
    num_experts = state_bank.num_experts
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in initial_params:
            continue
        if "lora" not in name.lower() or ".experts." not in name:
            continue
        init_val = initial_params[name].to(param.device)
        expert_idx = _extract_expert_index(name)
        if expert_idx is not None:
            coeff = weights[min(expert_idx, num_experts - 1)]
            term = coeff * torch.mean((param - init_val) ** 2)
        elif param.dim() > 0 and param.size(0) == num_experts:
            coeff = weights.to(param.device).view(num_experts, *([1] * (param.dim() - 1)))
            term = torch.mean(coeff * (param - init_val) ** 2)
        else:
            term = torch.mean((param - init_val) ** 2)
        reg = term if reg is None else reg + term
    if reg is None:
        return torch.tensor(0.0, device=state_bank.device)
    return reg


def apply_expert_gradient_scaling(model, state_bank: ExpertStateBank, sample_repr: torch.Tensor,
                                  sample_prior: torch.Tensor, risk: torch.Tensor,
                                  low_threshold: float,
                                  a_n: float, a_q: float, a_c: float,
                                  m_min: float, m_max: float):
    num_experts = state_bank.num_experts
    top2 = torch.topk(sample_prior, k=min(2, num_experts), dim=-1).indices
    novelty = torch.zeros(num_experts, device=sample_repr.device)
    has_novelty = torch.zeros(num_experts, dtype=torch.bool, device=sample_repr.device)
    for expert_idx in range(num_experts):
        assigned = (top2 == expert_idx).any(dim=-1) & (risk < low_threshold)
        if assigned.any():
            mean_repr = sample_repr[assigned].mean(dim=0)
            novelty[expert_idx] = torch.norm(mean_repr - state_bank.proto[expert_idx], p=2)
            has_novelty[expert_idx] = True
    scales = (
        a_n * novelty
        + a_q * (1.0 - state_bank.reliability)
        + a_c * (1.0 - state_bank.contamination)
    ).clamp(m_min, m_max)
    scales = torch.where(has_novelty, scales, torch.full_like(scales, m_min))

    for name, param in model.named_parameters():
        if param.grad is None or "lora" not in name.lower() or ".experts." not in name:
            continue
        expert_idx = _extract_expert_index(name)
        if expert_idx is not None:
            param.grad.mul_(scales[min(expert_idx, num_experts - 1)])
        elif param.grad.dim() > 0 and param.grad.size(0) == num_experts:
            view = scales.view(num_experts, *([1] * (param.grad.dim() - 1)))
            param.grad.mul_(view)


def apply_contamination_restoration(model, initial_params: Dict[str, torch.Tensor],
                                    state_bank: ExpertStateBank, rho: float):
    num_experts = state_bank.num_experts
    contamination = state_bank.contamination.detach()
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in initial_params:
            continue
        if "lora" not in name.lower() or ".experts." not in name:
            continue
        init_val = initial_params[name].to(param.device)
        expert_idx = _extract_expert_index(name)
        if expert_idx is not None:
            coeff = rho * contamination[min(expert_idx, num_experts - 1)]
            param.data = coeff * init_val + (1.0 - coeff) * param.data
        elif param.dim() > 0 and param.size(0) == num_experts:
            coeff = (rho * contamination).to(param.device).view(num_experts, *([1] * (param.dim() - 1)))
            param.data = coeff * init_val + (1.0 - coeff) * param.data


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
        gradient_checkpointing: Optional[bool] = True,
        target_layer: Optional[int] = 0,
        warmup_batches: Optional[int] = 10,
        tau_d: Optional[float] = 0.2,
        tau_i: Optional[float] = 0.2,
        tau_low: Optional[float] = 0.35,
        tau_high: Optional[float] = 0.65,
        lambda_r: Optional[float] = 2.0,
        lambda_bal: Optional[float] = 0.05,
        lambda_anc: Optional[float] = 1e-4,
        lambda_q: Optional[float] = 1.0,
        lambda_c: Optional[float] = 1.0,
        alpha_d: Optional[float] = 0.5,
        alpha_i: Optional[float] = 1.0,
        alpha_h: Optional[float] = 0.5,
        kappa: Optional[float] = 0.3,
        eta_e: Optional[float] = 0.9,
        eta_q: Optional[float] = 0.95,
        eta_c: Optional[float] = 0.95,
        safe_top_b: Optional[int] = 4,
        restoration_rho: Optional[float] = 0.1,
        a_n: Optional[float] = 0.5,
        a_q: Optional[float] = 0.25,
        a_c: Optional[float] = 0.25,
        m_min: Optional[float] = 0.1,
        m_max: Optional[float] = 1.0,
        max_examples_per_task: Optional[int] = 0,
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
        self.gradient_checkpointing = gradient_checkpointing
        self.target_layer = target_layer
        self.warmup_batches = warmup_batches
        self.tau_d = tau_d
        self.tau_i = tau_i
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.lambda_r = lambda_r
        self.lambda_bal = lambda_bal
        self.lambda_anc = lambda_anc
        self.lambda_q = lambda_q
        self.lambda_c = lambda_c
        self.alpha_d = alpha_d
        self.alpha_i = alpha_i
        self.alpha_h = alpha_h
        self.kappa = kappa
        self.eta_e = eta_e
        self.eta_q = eta_q
        self.eta_c = eta_c
        self.safe_top_b = safe_top_b
        self.restoration_rho = restoration_rho
        self.a_n = a_n
        self.a_q = a_q
        self.a_c = a_c
        self.m_min = m_min
        self.m_max = m_max
        self.max_examples_per_task = max_examples_per_task


def continual_test_time_adaptation_hera(args, model, tokenizer, optimizer, accelerator):
    model.eval()
    batch_size = args.eval_batch_size
    all_task_results = {}

    trainable_initial = collect_trainable_initial_params(model)
    if len(trainable_initial) == 0:
        raise ValueError(
            "No trainable parameters found. HERA requires trainable LoRA parameters. "
            "Please check the layer-selection diagnostics printed by module/moe_lora.py."
        )

    if isinstance(args.task, str):
        task_list = [task.strip() for task in args.task.split(",") if task.strip()]
    else:
        task_list = list(args.task)

    running_stats = RunningStatsBank()
    domain_tracker = DomainStateTracker(momentum=0.9)
    state_bank = None
    global_batch_step = 0

    for task_name in task_list:
        if accelerator.is_main_process:
            print(f"\n[HERA] Start task: {task_name}")

        task_dict = get_task_dict([task_name])
        task = task_dict[task_name]
        docs = list(task.validation_docs())
        if args.max_examples_per_task is not None and args.max_examples_per_task > 0:
            docs = docs[:args.max_examples_per_task]
        n_examples = len(docs)

        pre_acc_list = []
        memory_list = []
        time_list = []

        for i in range(0, n_examples, batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, n_examples)))
            samples_dict = {task_name: batch_indices}

            eval_model = accelerator.unwrap_model(model)
            lm_model = HFLM(
                pretrained=eval_model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                device=str(accelerator.device),
            )

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
                if accelerator.is_main_process:
                    print(f"[HERA][Warning] task {task_name} not in pre_results, skipping batch")
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
                model=eval_model,
                task_family=task_family,
                device=accelerator.device,
            )

            encoded = tokenizer(
                batch_contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = encoded["input_ids"].to(accelerator.device)
            attention_mask = encoded["attention_mask"].to(accelerator.device)

            probe_storage = {}
            probe_handles = register_probe_hooks(eval_model, args.target_layer, probe_storage)
            with torch.no_grad():
                probe_outputs = eval_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            for handle in probe_handles:
                handle.remove()

            if "pre_router_hidden" not in probe_storage or "router_logits" not in probe_storage:
                raise RuntimeError("Failed to capture router hidden states/logits for HERA.")

            sample_repr = sample_mean_hidden(
                probe_storage["pre_router_hidden"], attention_mask
            ).float()
            route_probs = mean_route_probs(
                probe_storage["router_logits"].float(), attention_mask
            ).float()
            pred_entropy = per_sample_prediction_entropy(
                probe_outputs.logits.float(), attention_mask
            )
            mean_pred_entropy = float(pred_entropy.mean().item())
            route_ent = route_entropy(route_probs)

            if state_bank is None:
                num_experts = route_probs.size(-1)
                hidden_dim = sample_repr.size(-1)
                state_bank = ExpertStateBank(num_experts=num_experts, hidden_dim=hidden_dim, device=accelerator.device)

            batch_repr = sample_repr.mean(dim=0)
            delta_t = domain_tracker.update(
                mu_t=batch_repr,
                rho_t=route_probs.mean(dim=0),
                mean_pred_entropy=mean_pred_entropy,
            )
            expert_repr = state_bank.blended_proto(delta_t=delta_t)
            proto_dist = prototype_mismatch(sample_repr, expert_repr)
            disagreement = expert_disagreement(route_probs)
            risk = compute_risk(pred_entropy, route_ent, disagreement, proto_dist, running_stats)

            if global_batch_step < args.warmup_batches:
                safe_mask = risk < args.tau_low
                state_bank.warmup_update(sample_repr, route_probs, safe_mask)
                global_batch_step += 1
                if global_batch_step == args.warmup_batches:
                    state_bank.ensure_initialized(sample_repr.mean(dim=0))
                if accelerator.is_main_process:
                    print(f"[HERA] Warm-up batch {global_batch_step}/{args.warmup_batches}")
                continue

            state_bank.ensure_initialized(sample_repr.mean(dim=0))

            sample_prior = build_sample_prior(
                sample_repr=sample_repr,
                batch_repr=batch_repr,
                expert_repr=expert_repr,
                state_bank=state_bank,
                risk=risk,
                tau_d=args.tau_d,
                tau_i=args.tau_i,
                alpha_d=args.alpha_d,
                alpha_i=args.alpha_i,
                alpha_h=args.alpha_h,
                lambda_q=args.lambda_q,
                lambda_c=args.lambda_c,
                safe_top_b=args.safe_top_b,
                tau_high=args.tau_high,
            )
            route_bias = args.lambda_r * torch.log(sample_prior.clamp_min(1e-6)).unsqueeze(1)

            start_time = time.time()
            model.train()
            optimizer.zero_grad()

            calib_handle = register_calibration_hook(model, args.target_layer, route_bias)
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                loss_per_sample = entropy_loss_per_sample(outputs.logits.float(), attention_mask)
                sample_weights = sample_update_weights(risk, args.tau_low, args.tau_high, args.kappa)
                weight_sum = sample_weights.sum().clamp_min(1e-12)
                loss_ent = torch.sum(sample_weights * loss_per_sample) / weight_sum
                loss_bal = balance_regularization(sample_prior, sample_weights)
                loss_anc = anchor_regularization(model, trainable_initial, state_bank)
                loss = loss_ent + args.lambda_anc * loss_anc

                accelerator.backward(loss)
            finally:
                calib_handle.remove()

            apply_expert_gradient_scaling(
                model=model,
                state_bank=state_bank,
                sample_repr=sample_repr.detach(),
                sample_prior=sample_prior.detach(),
                risk=risk.detach(),
                low_threshold=args.tau_low,
                a_n=args.a_n,
                a_q=args.a_q,
                a_c=args.a_c,
                m_min=args.m_min,
                m_max=args.m_max,
            )
            optimizer.step()
            apply_contamination_restoration(
                model=model,
                initial_params=trainable_initial,
                state_bank=state_bank,
                rho=args.restoration_rho,
            )
            state_bank.update_after_step(
                sample_repr=sample_repr.detach(),
                sample_prior=sample_prior.detach(),
                risk=risk.detach(),
                low_threshold=args.tau_low,
                high_threshold=args.tau_high,
                eta_e=args.eta_e,
                eta_q=args.eta_q,
                eta_c=args.eta_c,
            )

            if torch.cuda.is_available():
                memory_list.append(torch.cuda.memory_allocated(accelerator.device) / 1024 ** 3)
            time_list.append((time.time() - start_time) / max(input_ids.size(0), 1))

            del probe_outputs, outputs, loss, loss_ent, loss_bal, loss_anc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.eval()
            global_batch_step += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)
            torch.cuda.empty_cache()

        summary = {
            "task": task_name,
            "pre_acc_mean": sum(pre_acc_list) / len(pre_acc_list) if pre_acc_list else 0.0,
            "cuda_memory_gb": sum(memory_list) / len(memory_list) if memory_list else 0.0,
            "time_per_sample": sum(time_list) / len(time_list) if time_list else 0.0,
        }
        if accelerator.is_main_process:
            print(f"[HERA] Task {task_name} summary: {summary}")
        all_task_results[task_name] = summary

    return all_task_results


def run_hera_opt(
    task: str,
    model_name: Optional[str] = "./Qwen1.5-MoE-A2.7B-Chat",
    eval_batch_size: Optional[int] = None,
    result_path: Optional[str] = "results/hera_results.txt",
    lr: Optional[float] = 1e-5,
    lora_r: Optional[int] = 16,
    lora_alpha: Optional[int] = 16,
    gradient_checkpointing: Optional[bool] = True,
    target_layer: Optional[int] = 0,
    warmup_batches: Optional[int] = 10,
    tau_d: Optional[float] = 0.2,
    tau_i: Optional[float] = 0.2,
    tau_low: Optional[float] = 0.35,
    tau_high: Optional[float] = 0.65,
    lambda_r: Optional[float] = 2.0,
    lambda_bal: Optional[float] = 0.05,
    lambda_anc: Optional[float] = 1e-4,
    lambda_q: Optional[float] = 1.0,
    lambda_c: Optional[float] = 1.0,
    alpha_d: Optional[float] = 0.5,
    alpha_i: Optional[float] = 1.0,
    alpha_h: Optional[float] = 0.5,
    kappa: Optional[float] = 0.3,
    eta_e: Optional[float] = 0.9,
    eta_q: Optional[float] = 0.95,
    eta_c: Optional[float] = 0.95,
    safe_top_b: Optional[int] = 4,
    restoration_rho: Optional[float] = 0.1,
    a_n: Optional[float] = 0.5,
    a_q: Optional[float] = 0.25,
    a_c: Optional[float] = 0.25,
    m_min: Optional[float] = 0.1,
    m_max: Optional[float] = 1.0,
    max_examples_per_task: Optional[int] = 0,
):
    torch.manual_seed(0)
    accelerator = Accelerator()
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = int(eval_batch_size) if eval_batch_size is not None else infer_default_eval_batch_size(model_name)

    args = Args(
        task=task,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        result_path=result_path,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        gradient_checkpointing=gradient_checkpointing,
        target_layer=target_layer,
        warmup_batches=warmup_batches,
        tau_d=tau_d,
        tau_i=tau_i,
        tau_low=tau_low,
        tau_high=tau_high,
        lambda_r=lambda_r,
        lambda_bal=lambda_bal,
        lambda_anc=lambda_anc,
        lambda_q=lambda_q,
        lambda_c=lambda_c,
        alpha_d=alpha_d,
        alpha_i=alpha_i,
        alpha_h=alpha_h,
        kappa=kappa,
        eta_e=eta_e,
        eta_q=eta_q,
        eta_c=eta_c,
        safe_top_b=safe_top_b,
        restoration_rho=restoration_rho,
        a_n=a_n,
        a_q=a_q,
        a_c=a_c,
        m_min=m_min,
        m_max=m_max,
        max_examples_per_task=max_examples_per_task,
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
    )
    if hasattr(model, "config") and model.config is not None:
        model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.use_cache = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = inject_lora_into_moe(model, r=args.lora_r, alpha=args.lora_alpha)
    freeze_non_lora_params(model, layers_to_use=[args.target_layer])

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        print("========== Starting HERA Evaluation ==========")
        print(f"Start Time:   {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model Path:   {args.model_name}")
        print(f"Task:         {task}")
        print(f"Batch Size:   {args.eval_batch_size}")
        print(f"LR:           {args.lr}")
        print(f"LoRA (r/a):   {args.lora_r}/{args.lora_alpha}")
        print(f"Target Layer: {args.target_layer}")
        print(f"Warm-up:      {args.warmup_batches}")
        print(f"Max Examples: {args.max_examples_per_task}")
        print(f"Grad Ckpt:    {args.gradient_checkpointing}")
        print(f"Device:       {accelerator.device}")
        print("==============================================")

    start_ts = time.time()
    results = continual_test_time_adaptation_hera(args, model, tokenizer, optimizer, accelerator)
    total_seconds = time.time() - start_ts
    end_dt = datetime.now()

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    else:
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0

    if accelerator.is_main_process:
        print(
            f"[HERA] Results: {results} | Time: {total_seconds:.2f} seconds | "
            f"Peak Alloc: {peak_allocated_gb:.2f} GB | Peak Reserved: {peak_reserved_gb:.2f} GB"
        )

        if args.result_path is not None:
            result_dir = os.path.dirname(args.result_path)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
            with open(args.result_path, "w", encoding="utf-8") as f:
                f.write("HERA Evaluation Results\n")
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
                f.write(f"Target Layer: {args.target_layer}\n")
                f.write(f"Warm-up Batches: {args.warmup_batches}\n")
                f.write(f"Max Examples Per Task: {args.max_examples_per_task}\n")
                f.write(f"Gradient Checkpointing: {args.gradient_checkpointing}\n")
                f.write("=" * 50 + "\n")
                for task_name, summary in results.items():
                    f.write(f"{task_name}: {summary}\n")


if __name__ == "__main__":
    Fire(run_hera_opt)
