import gc
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict

from module.Construct_context import construct_context
from module.hera_risk import (
    HeraRiskEstimator,
    approximate_expert_disagreement,
    predictive_entropy_from_logits,
    prototype_mismatch,
    routing_entropy_from_sample_probs,
)
from module.hera_router import HeraRouterCalibrator
from module.hera_state_bank import ExpertStateBank, HeraStateConfig


@dataclass
class HeraConfig:
    warmup_batches: int = 10
    tau_low: float = 0.35
    tau_high: float = 0.65
    kappa: float = 0.3
    safe_top_b: int = 4

    tau_d: float = 0.2
    tau_i: float = 0.2
    alpha_d: float = 0.5
    alpha_i: float = 1.0
    alpha_h: float = 0.5
    lambda_q: float = 1.0
    lambda_c: float = 1.0
    lambda_router: float = 2.0

    lambda_rho: float = 1.0
    lambda_h: float = 1.0
    lambda_bal: float = 0.05
    lambda_anchor: float = 1e-4
    restore_rho: float = 0.1

    a_n: float = 0.5
    a_q: float = 0.25
    a_c: float = 0.25
    m_min: float = 0.1
    m_max: float = 1.0


def _cfg_from_args(args) -> HeraConfig:
    cfg = HeraConfig()
    for key in cfg.__dict__.keys():
        if hasattr(args, key):
            setattr(cfg, key, getattr(args, key))
    return cfg


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
        v = task_result.get("exact_match,none", 0.0)
        return v, v
    raise ValueError(f"Unknown OUTPUT_TYPE: {task.OUTPUT_TYPE}")


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    pooled = torch.sum(hidden * mask, dim=1) / (torch.sum(mask, dim=1) + 1e-10)
    pooled = F.layer_norm(pooled, (pooled.size(-1),))
    return F.normalize(pooled, p=2, dim=-1)


def _extract_router_probs(outputs, layer_idx: int, batch_size: int):
    router_logits = outputs.router_logits[layer_idx]
    if router_logits.dim() == 2:
        num_experts = router_logits.size(-1)
        seq_len = max(1, router_logits.size(0) // max(1, batch_size))
        router_logits = router_logits.view(batch_size, seq_len, num_experts)
    elif router_logits.dim() != 3:
        raise ValueError(f"Unsupported router logits shape: {tuple(router_logits.shape)}")

    route_probs = torch.softmax(router_logits, dim=-1)
    sample_route_probs = torch.mean(route_probs, dim=1)
    return router_logits, route_probs, sample_route_probs


def _capture_initial_params(model, layers_to_use: List[int]) -> Dict[str, torch.Tensor]:
    init = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".layers." not in name:
            continue
        try:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
        except ValueError:
            continue
        if layer_idx not in layers_to_use:
            continue
        if "lora" in name.lower() or ".mlp.gate." in name:
            init[name] = p.detach().clone().cpu()
    return init


def _build_expert_param_map(model, layers_to_use: List[int]) -> Dict[int, List[torch.nn.Parameter]]:
    mapping: Dict[int, List[torch.nn.Parameter]] = {}
    pattern = re.compile(r"layers\.(\d+)\..*?experts\.(\d+)\..*lora", re.IGNORECASE)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = pattern.search(name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        expert_idx = int(m.group(2))
        if layer_idx not in layers_to_use:
            continue
        mapping.setdefault(expert_idx, []).append(p)
    return mapping


def _scale_expert_gradients(expert_param_map: Dict[int, List[torch.nn.Parameter]], lr_mult: torch.Tensor):
    for expert_idx, params in expert_param_map.items():
        scale = float(lr_mult[expert_idx].item())
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)


def _apply_restore(
    expert_param_map: Dict[int, List[torch.nn.Parameter]],
    init_params: Dict[str, torch.Tensor],
    contamination: torch.Tensor,
    restore_rho: float,
):
    if len(expert_param_map) == 0:
        return
    for expert_idx, params in expert_param_map.items():
        alpha = float(restore_rho * contamination[expert_idx].item())
        if alpha <= 0.0:
            continue
        for p in params:
            if not hasattr(p, "_hera_param_name"):
                continue
            name = p._hera_param_name
            if name not in init_params:
                continue
            init = init_params[name].to(p.device)
            p.data = alpha * init + (1.0 - alpha) * p.data


def continual_test_time_adaptation_hera(
    args,
    model,
    tokenizer,
    layers_to_use: Optional[List[int]] = None,
    device=None,
):
    cfg = _cfg_from_args(args)
    if layers_to_use is None:
        layers_to_use = [0]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    batch_size = args.eval_batch_size
    lm_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device_map="auto")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    init_params = _capture_initial_params(model, layers_to_use)
    expert_param_map = _build_expert_param_map(model, layers_to_use)
    for name, p in model.named_parameters():
        if p.requires_grad:
            p._hera_param_name = name

    risk_estimator = HeraRiskEstimator(weights=(1.0, 0.5, 0.75, 1.0))
    all_task_results = {}

    if isinstance(args.task, str):
        task_list = [x.strip() for x in args.task.split(",") if x.strip()]
    else:
        task_list = args.task

    for task_name in task_list:
        print(f"\n[HERA] Start task: {task_name}")
        task_dict = get_task_dict([task_name])
        task = task_dict[task_name]
        docs = list(task.validation_docs())
        n_examples = len(docs)

        pre_acc_list = []
        memory_list = []
        time_list = []

        state_bank = None
        warmup_done_at = -1

        for i in range(0, n_examples, batch_size):
            batch_docs = docs[i : i + batch_size]
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
                print(f"[HERA][Warning] task {task_name} not in pre_results, skip batch")
                continue
            pre_acc, _ = get_main_metric(task, pre_task_result)
            pre_acc_list.append(pre_acc)

            start_time = time.time()

            requests_list = []
            _ = get_task_family(task)
            for doc in batch_docs:
                ctx = task.fewshot_context(doc, num_fewshot=0)
                requests_list.append(task.construct_requests(doc, ctx))

            batch_contexts = construct_context(
                requests_list=requests_list,
                tokenizer=tokenizer,
                model=model,
                task_family=get_task_family(task),
                device=device,
            )

            encoded = tokenizer(batch_contexts, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs_probe = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_router_logits=True,
                    output_hidden_states=True,
                )

            if getattr(outputs_probe, "router_logits", None) is None:
                print("[HERA][Warning] Model output has no router_logits, skip batch")
                continue

            layer_idx = layers_to_use[0]
            _, route_probs, sample_route_probs = _extract_router_probs(outputs_probe, layer_idx, input_ids.size(0))

            hs_idx = min(layer_idx + 1, len(outputs_probe.hidden_states) - 1)
            sample_embs = _mean_pool(outputs_probe.hidden_states[hs_idx], attention_mask)

            pred_entropy = predictive_entropy_from_logits(outputs_probe.logits, attention_mask)
            route_entropy = routing_entropy_from_sample_probs(sample_route_probs)
            disagreement = approximate_expert_disagreement(route_probs, attention_mask)

            if state_bank is None:
                num_experts = sample_route_probs.size(-1)
                state_bank = ExpertStateBank(
                    num_experts=num_experts,
                    hidden_dim=sample_embs.size(-1),
                    device=device,
                    config=HeraStateConfig(warmup_batches=cfg.warmup_batches),
                )

            batch_mu = sample_embs.mean(dim=0)
            batch_rho = sample_route_probs.mean(dim=0)
            batch_entropy = pred_entropy.mean()
            state_bank.update_domain_state(batch_mu, batch_rho, batch_entropy, cfg.lambda_rho, cfg.lambda_h)

            proto_now = state_bank.get_fused_prototypes()
            mismatch = prototype_mismatch(sample_embs, proto_now)
            risk = risk_estimator(pred_entropy, route_entropy, disagreement, mismatch)

            if not state_bank.warmup_done:
                state_bank.warmup_update(sample_embs, sample_route_probs, risk, cfg.tau_low)
                if state_bank._warmup_seen_batches >= state_bank.warmup_batches:
                    state_bank.finalize_warmup(global_mean=batch_mu)
                    warmup_done_at = i
                time_list.append((time.time() - start_time) / max(1, input_ids.size(0)))
                if torch.cuda.is_available():
                    memory_list.append(torch.cuda.memory_allocated() / 1024 ** 3)
                continue

            domain_prior = state_bank.compute_domain_prior(batch_mu, cfg.tau_d)
            instance_prior = state_bank.compute_instance_prior(sample_embs, cfg.tau_i)
            health_prior = state_bank.compute_health_prior(cfg.lambda_q, cfg.lambda_c)
            sample_prior = state_bank.combine_priors(
                domain_prior,
                instance_prior,
                health_prior,
                cfg.alpha_d,
                cfg.alpha_i,
                cfg.alpha_h,
            )
            sample_prior = state_bank.apply_safe_mask(sample_prior, risk, cfg.tau_high, cfg.safe_top_b)

            calibrator = HeraRouterCalibrator(model=model, layer_idx=layer_idx, lam=cfg.lambda_router, eps=1e-6)
            handle = calibrator.attach(sample_prior, batch_size=input_ids.size(0))
            model.train()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            handle.remove()

            probs = torch.softmax(outputs.logits, dim=-1)
            token_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
            sample_entropy = torch.sum(token_entropy * attention_mask.float(), dim=-1) / (
                torch.sum(attention_mask.float(), dim=-1) + 1e-10
            )

            weights = torch.where(
                risk < cfg.tau_low,
                torch.ones_like(risk),
                torch.where(risk < cfg.tau_high, torch.full_like(risk, cfg.kappa), torch.zeros_like(risk)),
            )
            loss_ent = torch.sum(weights * sample_entropy) / (torch.sum(weights) + 1e-10)

            weighted_prior = sample_prior * weights.unsqueeze(-1)
            prior_mean = weighted_prior.sum(dim=0) / (weights.sum() + 1e-10)
            uniform = torch.full_like(prior_mean, 1.0 / prior_mean.numel())
            loss_bal = F.kl_div(torch.log(prior_mean + 1e-12), uniform, reduction="sum")

            loss_anchor = torch.tensor(0.0, device=device)
            health_scalar = torch.clamp(state_bank.reliability.mean() + state_bank.contamination.mean(), 0.0, 2.0)
            for name, p in model.named_parameters():
                if p.requires_grad and name in init_params:
                    loss_anchor = loss_anchor + torch.norm(p - init_params[name].to(p.device), p=2) ** 2

            loss = loss_ent + cfg.lambda_bal * loss_bal + cfg.lambda_anchor * health_scalar * loss_anchor

            optimizer.zero_grad()
            loss.backward()

            novelty = state_bank.compute_novelty(sample_embs.detach(), sample_prior.detach(), risk.detach(), cfg.tau_low)
            lr_mult = state_bank.compute_lr_multiplier(
                novelty,
                a_n=cfg.a_n,
                a_q=cfg.a_q,
                a_c=cfg.a_c,
                m_min=cfg.m_min,
                m_max=cfg.m_max,
            )

            if len(expert_param_map) > 0:
                _scale_expert_gradients(expert_param_map, lr_mult)
            else:
                global_scale = float(lr_mult.mean().item())
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.mul_(global_scale)

            optimizer.step()
            model.eval()

            _apply_restore(expert_param_map, init_params, state_bank.contamination, cfg.restore_rho)

            state_bank.update_after_step(sample_embs.detach(), sample_prior.detach(), risk.detach(), cfg.tau_low, cfg.tau_high)

            time_list.append((time.time() - start_time) / max(1, input_ids.size(0)))
            if torch.cuda.is_available():
                memory_list.append(torch.cuda.memory_allocated() / 1024 ** 3)

            del outputs_probe, outputs, probs, token_entropy, sample_entropy, loss
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        summary = {
            "task": task_name,
            "pre_acc_mean": sum(pre_acc_list) / len(pre_acc_list) if pre_acc_list else 0.0,
            "cuda_memory_gb": sum(memory_list) / len(memory_list) if memory_list else 0.0,
            "time_per_sample": sum(time_list) / len(time_list) if time_list else 0.0,
            "warmup_done_batch_start": warmup_done_at,
        }
        print(f"[HERA] Task {task_name} summary: {summary}")
        all_task_results[task_name] = summary

    return all_task_results
