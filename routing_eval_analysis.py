import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from fire import Fire
from lm_eval.tasks import get_task_dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from hera_moe_opt import (
    DomainStateTracker,
    ExpertStateBank,
    RunningStatsBank,
    build_sample_prior,
    construct_context,
    cosine_prior,
    entropy_loss_per_sample,
    expert_disagreement,
    infer_default_eval_batch_size,
    mean_route_probs,
    per_sample_prediction_entropy,
    prototype_mismatch,
    register_calibration_hook,
    register_probe_hooks,
    route_entropy,
    sample_mean_hidden,
)
from module.moe_lora import freeze_non_lora_params, inject_lora_into_moe


def safe_item(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    return value


def topk_counts(P: torch.Tensor, top_k: int) -> torch.Tensor:
    _, num_experts = P.shape
    topk = torch.topk(P, k=min(top_k, num_experts), dim=-1).indices
    counts = torch.zeros(num_experts, device=P.device, dtype=torch.float32)
    for k in range(topk.size(1)):
        counts.scatter_add_(
            0,
            topk[:, k],
            torch.ones_like(topk[:, k], dtype=counts.dtype),
        )
    return counts


def load_cv(P: torch.Tensor, top_k: int = 2, eps: float = 1e-12) -> torch.Tensor:
    counts = topk_counts(P, top_k=top_k)
    return counts.std(unbiased=False) / counts.mean().clamp_min(eps)


def mri(P: torch.Tensor, top_k: int = 2, eps: float = 1e-12) -> torch.Tensor:
    counts = topk_counts(P, top_k=top_k)
    usage = counts / counts.sum().clamp_min(eps)
    return usage.max()


def domain_purity(
    P: torch.Tensor,
    domain_ids: torch.Tensor,
    num_domains: int,
    top_k: int = 2,
    eps: float = 1e-12,
) -> torch.Tensor:
    _, num_experts = P.shape
    topk = torch.topk(P, k=min(top_k, num_experts), dim=-1).indices
    counts = torch.zeros(num_experts, num_domains, device=P.device, dtype=torch.float32)

    for k in range(topk.size(1)):
        experts = topk[:, k]
        counts.index_put_(
            (experts, domain_ids),
            torch.ones(domain_ids.size(0), device=P.device, dtype=counts.dtype),
            accumulate=True,
        )

    expert_total = counts.sum(dim=-1)
    active = expert_total > 0
    if not active.any():
        return torch.tensor(0.0, device=P.device)

    prob = counts[active] / expert_total[active].unsqueeze(-1).clamp_min(eps)
    entropy = -(prob * torch.log(prob + eps)).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(max(num_domains, 2)), device=P.device))
    purity = 1.0 - entropy / max_entropy.clamp_min(eps)
    return purity.mean()


def sim_consistency(P: torch.Tensor, sample_repr: torch.Tensor, top_k: int = 2) -> torch.Tensor:
    num_samples, num_experts = P.shape
    if num_samples <= 1:
        return torch.tensor(0.0, device=P.device)

    K = min(top_k, num_experts)
    x = F.normalize(sample_repr.float(), dim=-1)
    sim = x @ x.T
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=-1)

    topk = torch.topk(P, k=K, dim=-1).indices
    topk_nn = topk[nn_idx]
    overlap = (topk.unsqueeze(2) == topk_nn.unsqueeze(1)).any(dim=2)
    return (overlap.float().sum(dim=1) / float(K)).mean()


def oracle_hit(P: torch.Tensor, oracle_expert: torch.Tensor, top_k: int = 2) -> torch.Tensor:
    topk = torch.topk(P, k=min(top_k, P.size(-1)), dim=-1).indices
    return (topk == oracle_expert.unsqueeze(-1)).any(dim=-1).float().mean()


def routing_margin(P: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(P, k=min(2, P.size(-1)), dim=-1).values
    if top2.size(-1) < 2:
        return torch.tensor(0.0, device=P.device)
    return (top2[:, 0] - top2[:, 1]).mean()


def routing_margin_by_risk(
    P: torch.Tensor,
    risk: torch.Tensor,
    tau_low: float = 0.35,
    tau_high: float = 0.65,
) -> Dict[str, torch.Tensor]:
    top2 = torch.topk(P, k=min(2, P.size(-1)), dim=-1).values
    if top2.size(-1) < 2:
        zero = torch.tensor(0.0, device=P.device)
        return {"margin_low": zero, "margin_mid": zero, "margin_high": zero}

    margin = top2[:, 0] - top2[:, 1]
    low = risk < tau_low
    mid = (risk >= tau_low) & (risk < tau_high)
    high = risk >= tau_high

    def masked_mean(mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=P.device)
        return margin[mask].mean()

    return {
        "margin_low": masked_mean(low),
        "margin_mid": masked_mean(mid),
        "margin_high": masked_mean(high),
    }


def evaluate_routing_metrics(
    P: torch.Tensor,
    sample_repr: torch.Tensor,
    domain_ids: torch.Tensor,
    num_domains: int,
    risk: Optional[torch.Tensor] = None,
    oracle_expert_ids: Optional[torch.Tensor] = None,
    top_k: int = 2,
    prefix: str = "",
) -> Dict[str, float]:
    prefix = f"{prefix}_" if prefix else ""
    results = {
        f"{prefix}load_cv": safe_item(load_cv(P, top_k=top_k)),
        f"{prefix}mri": safe_item(mri(P, top_k=top_k)),
        f"{prefix}domain_purity": safe_item(
            domain_purity(P, domain_ids, num_domains=num_domains, top_k=top_k)
        ),
        f"{prefix}sim_consistency": safe_item(sim_consistency(P, sample_repr, top_k=top_k)),
        f"{prefix}routing_margin": safe_item(routing_margin(P)),
    }
    if risk is not None:
        results.update(
            {
                f"{prefix}{key}": safe_item(value)
                for key, value in routing_margin_by_risk(P, risk).items()
            }
        )
    if oracle_expert_ids is not None:
        results[f"{prefix}oracle_hit"] = safe_item(
            oracle_hit(P, oracle_expert_ids, top_k=top_k)
        )
    return results


def summarize_batch_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "max": float(tensor.max().item()),
    }


def make_proto_only_prior(sample_repr: torch.Tensor, expert_repr: torch.Tensor, tau_i: float) -> torch.Tensor:
    return cosine_prior(sample_repr, expert_repr, tau_i)


def make_proto_health_prior(
    sample_repr: torch.Tensor,
    expert_repr: torch.Tensor,
    state_bank: ExpertStateBank,
    tau_i: float,
    alpha_i: float,
    alpha_h: float,
    lambda_q: float,
    lambda_c: float,
) -> torch.Tensor:
    proto = cosine_prior(sample_repr, expert_repr, tau_i)
    health = state_bank.health_prior(lambda_q=lambda_q, lambda_c=lambda_c)
    prior = proto.pow(alpha_i) * health.unsqueeze(0).pow(alpha_h)
    return prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def mean_sequence_entropy(logits: torch.Tensor, attention_mask: torch.Tensor) -> float:
    return float(entropy_loss_per_sample(logits.float(), attention_mask).mean().item())


def build_one_hot_bias(num_experts: int, expert_idx: int, scale: float, eps: float = 1e-6) -> torch.Tensor:
    prior = torch.full((1, num_experts), eps, dtype=torch.float32)
    prior[0, expert_idx] = 1.0
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(eps)
    return scale * torch.log(prior.clamp_min(eps))


def compute_oracle_expert_ids(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    P_ori: torch.Tensor,
    P_hera: torch.Tensor,
    target_layer: int,
    top_l: int,
    bias_scale: float,
) -> torch.Tensor:
    num_samples, num_experts = P_ori.shape
    candidates_ori = torch.topk(P_ori, k=min(top_l, num_experts), dim=-1).indices
    candidates_hera = torch.topk(P_hera, k=min(top_l, num_experts), dim=-1).indices
    oracle_ids = []

    for idx in range(num_samples):
        merged = torch.cat([candidates_ori[idx], candidates_hera[idx]], dim=0).unique(sorted=False)
        best_entropy = None
        best_expert = int(merged[0].item())
        sample_ids = input_ids[idx:idx + 1]
        sample_mask = attention_mask[idx:idx + 1]

        for expert_idx in merged.tolist():
            bias = build_one_hot_bias(num_experts, expert_idx, bias_scale).to(sample_ids.device)
            handle = register_calibration_hook(model, target_layer, bias)
            try:
                with torch.no_grad():
                    outputs = model(
                        input_ids=sample_ids,
                        attention_mask=sample_mask,
                        use_cache=False,
                    )
                entropy_value = mean_sequence_entropy(outputs.logits, sample_mask)
            finally:
                handle.remove()

            if best_entropy is None or entropy_value < best_entropy:
                best_entropy = entropy_value
                best_expert = expert_idx

        oracle_ids.append(best_expert)

    return torch.tensor(oracle_ids, device=input_ids.device, dtype=torch.long)


@dataclass
class RoutingEvalArgs:
    task: str
    model_name: str = "./Qwen1.5-MoE-A2.7B-Chat"
    eval_batch_size: Optional[int] = None
    output_path: str = "results/routing_eval"
    lora_r: int = 16
    lora_alpha: int = 16
    target_layer: int = 0
    warmup_batches: int = 10
    max_examples_per_task: Optional[int] = 200
    top_k: int = 2
    top_l_oracle: int = 4
    tau_d: float = 0.2
    tau_i: float = 0.2
    tau_low: float = 0.35
    tau_high: float = 0.65
    lambda_r: float = 2.0
    lambda_q: float = 1.0
    lambda_c: float = 1.0
    alpha_d: float = 0.5
    alpha_i: float = 1.0
    alpha_h: float = 0.5
    safe_top_b: int = 4
    compute_oracle_hit: bool = False


def run_routing_eval(
    task: str,
    model_name: str = "./Qwen1.5-MoE-A2.7B-Chat",
    eval_batch_size: Optional[int] = None,
    output_path: str = "results/routing_eval",
    lora_r: int = 16,
    lora_alpha: int = 16,
    target_layer: int = 0,
    warmup_batches: int = 10,
    max_examples_per_task: Optional[int] = 200,
    top_k: int = 2,
    top_l_oracle: int = 4,
    tau_d: float = 0.2,
    tau_i: float = 0.2,
    tau_low: float = 0.35,
    tau_high: float = 0.65,
    lambda_r: float = 2.0,
    lambda_q: float = 1.0,
    lambda_c: float = 1.0,
    alpha_d: float = 0.5,
    alpha_i: float = 1.0,
    alpha_h: float = 0.5,
    safe_top_b: int = 4,
    compute_oracle_hit: bool = False,
):
    args = RoutingEvalArgs(
        task=task,
        model_name=model_name,
        eval_batch_size=(
            infer_default_eval_batch_size(model_name)
            if eval_batch_size is None else eval_batch_size
        ),
        output_path=output_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_layer=target_layer,
        warmup_batches=warmup_batches,
        max_examples_per_task=max_examples_per_task,
        top_k=top_k,
        top_l_oracle=top_l_oracle,
        tau_d=tau_d,
        tau_i=tau_i,
        tau_low=tau_low,
        tau_high=tau_high,
        lambda_r=lambda_r,
        lambda_q=lambda_q,
        lambda_c=lambda_c,
        alpha_d=alpha_d,
        alpha_i=alpha_i,
        alpha_h=alpha_h,
        safe_top_b=safe_top_b,
        compute_oracle_hit=compute_oracle_hit,
    )

    os.makedirs(args.output_path, exist_ok=True)

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
        model.generation_config.use_cache = False
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    model = inject_lora_into_moe(model, r=args.lora_r, alpha=args.lora_alpha)
    freeze_non_lora_params(model, layers_to_use=[args.target_layer])
    model.eval()

    model_device = next(model.parameters()).device

    if isinstance(args.task, str):
        task_list = [item.strip() for item in args.task.split(",") if item.strip()]
    else:
        task_list = list(args.task)
    task_to_domain = {task_name: idx for idx, task_name in enumerate(task_list)}
    task_docs_map = {}
    total_batches = 0
    for task_name in task_list:
        task_dict = get_task_dict([task_name])
        task_obj = task_dict[task_name]
        docs = list(task_obj.validation_docs())
        if args.max_examples_per_task is not None and args.max_examples_per_task > 0:
            docs = docs[: args.max_examples_per_task]
        task_docs_map[task_name] = docs
        total_batches += (len(docs) + args.eval_batch_size - 1) // args.eval_batch_size

    running_stats = RunningStatsBank()
    domain_tracker = DomainStateTracker(momentum=0.9)
    state_bank = None
    global_batch_step = 0

    logs = {
        "task_name": [],
        "domain_id": [],
        "P_ori": [],
        "P_proto": [],
        "P_proto_health": [],
        "P_hera": [],
        "sample_repr": [],
        "risk": [],
    }
    oracle_logs = {
        "proto": [],
        "proto_health": [],
        "hera": [],
    }
    batch_stats = {
        "ori_load_cv": [],
        "ori_mri": [],
        "proto_load_cv": [],
        "proto_mri": [],
        "proto_health_load_cv": [],
        "proto_health_mri": [],
        "hera_load_cv": [],
        "hera_mri": [],
    }

    print("========== Routing Evaluation ==========")
    print(f"Start Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Path:     {args.model_name}")
    print(f"Tasks:          {','.join(task_list)}")
    print(f"Batch Size:     {args.eval_batch_size}")
    print(f"Warm-up Batches:{args.warmup_batches}")
    print(f"Top-K:          {args.top_k}")
    print(f"Oracle Hit:     {args.compute_oracle_hit}")
    print("========================================")

    overall_pbar = tqdm(total=total_batches, desc="RoutingEval Total", dynamic_ncols=True)

    for task_idx, task_name in enumerate(task_list, start=1):
        task_dict = get_task_dict([task_name])
        task_obj = task_dict[task_name]
        docs = task_docs_map[task_name]

        print(f"\n[RoutingEval] Task: {task_name} | Examples: {len(docs)}")
        task_total_batches = (len(docs) + args.eval_batch_size - 1) // args.eval_batch_size
        task_pbar = tqdm(
            total=task_total_batches,
            desc=f"{task_name} ({task_idx}/{len(task_list)})",
            dynamic_ncols=True,
            leave=False,
        )

        for start in range(0, len(docs), args.eval_batch_size):
            batch_docs = docs[start:start + args.eval_batch_size]
            requests_list = []

            for doc in batch_docs:
                ctx = task_obj.fewshot_context(doc, num_fewshot=0)
                requests = task_obj.construct_requests(doc, ctx)
                requests_list.append(requests)

            batch_contexts = construct_context(
                requests_list=requests_list,
                tokenizer=tokenizer,
                model=model,
                task_family=None,
                device=model_device,
            )
            if not batch_contexts:
                continue

            encoded = tokenizer(
                batch_contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = encoded["input_ids"].to(model_device)
            attention_mask = encoded["attention_mask"].to(model_device)

            probe_storage = {}
            handles = register_probe_hooks(model, args.target_layer, probe_storage)
            try:
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
            finally:
                for handle in handles:
                    handle.remove()

            if "pre_router_hidden" not in probe_storage or "router_logits" not in probe_storage:
                raise RuntimeError("Failed to capture router probe tensors.")

            sample_repr = sample_mean_hidden(
                probe_storage["pre_router_hidden"], attention_mask
            ).float()
            route_probs = mean_route_probs(
                probe_storage["router_logits"].float(), attention_mask
            ).float()
            pred_entropy = per_sample_prediction_entropy(
                outputs.logits.float(), attention_mask
            )
            route_ent = route_entropy(route_probs)
            batch_repr = sample_repr.mean(dim=0)

            if state_bank is None:
                state_bank = ExpertStateBank(
                    num_experts=route_probs.size(-1),
                    hidden_dim=sample_repr.size(-1),
                    device=sample_repr.device,
                )

            delta_t = domain_tracker.update(
                mu_t=batch_repr,
                rho_t=route_probs.mean(dim=0),
                mean_pred_entropy=float(pred_entropy.mean().item()),
            )
            expert_repr = state_bank.blended_proto(delta_t=delta_t)
            proto_dist = prototype_mismatch(sample_repr, expert_repr)
            disagreement = expert_disagreement(route_probs)
            risk = compute_risk = None
            risk = torch.sigmoid(
                1.0 * running_stats.normalize("pred_entropy", pred_entropy)
                + 0.5 * running_stats.normalize("route_entropy", route_ent)
                + 0.75 * running_stats.normalize("disagreement", disagreement)
                + 1.0 * running_stats.normalize("proto_dist", proto_dist)
            )

            if global_batch_step < args.warmup_batches:
                state_bank.warmup_update(sample_repr, route_probs, risk < args.tau_low)
                global_batch_step += 1
                if global_batch_step == args.warmup_batches:
                    state_bank.ensure_initialized(sample_repr.mean(dim=0))
                task_pbar.set_postfix_str(
                    f"warmup={global_batch_step}/{args.warmup_batches} logged={len(logs['task_name'])}"
                )
                task_pbar.update(1)
                overall_pbar.update(1)
                continue

            state_bank.ensure_initialized(sample_repr.mean(dim=0))
            expert_repr = state_bank.blended_proto(delta_t=delta_t)

            P_proto = make_proto_only_prior(sample_repr, expert_repr, tau_i=args.tau_i)
            P_proto_health = make_proto_health_prior(
                sample_repr=sample_repr,
                expert_repr=expert_repr,
                state_bank=state_bank,
                tau_i=args.tau_i,
                alpha_i=args.alpha_i,
                alpha_h=args.alpha_h,
                lambda_q=args.lambda_q,
                lambda_c=args.lambda_c,
            )
            P_hera = build_sample_prior(
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

            logs["task_name"].extend([task_name] * sample_repr.size(0))
            logs["domain_id"].append(
                torch.full(
                    (sample_repr.size(0),),
                    task_to_domain[task_name],
                    dtype=torch.long,
                )
            )
            logs["P_ori"].append(route_probs.detach().cpu())
            logs["P_proto"].append(P_proto.detach().cpu())
            logs["P_proto_health"].append(P_proto_health.detach().cpu())
            logs["P_hera"].append(P_hera.detach().cpu())
            logs["sample_repr"].append(sample_repr.detach().cpu())
            logs["risk"].append(risk.detach().cpu())

            batch_stats["ori_load_cv"].append(float(load_cv(route_probs, args.top_k).item()))
            batch_stats["ori_mri"].append(float(mri(route_probs, args.top_k).item()))
            batch_stats["proto_load_cv"].append(float(load_cv(P_proto, args.top_k).item()))
            batch_stats["proto_mri"].append(float(mri(P_proto, args.top_k).item()))
            batch_stats["proto_health_load_cv"].append(
                float(load_cv(P_proto_health, args.top_k).item())
            )
            batch_stats["proto_health_mri"].append(
                float(mri(P_proto_health, args.top_k).item())
            )
            batch_stats["hera_load_cv"].append(float(load_cv(P_hera, args.top_k).item()))
            batch_stats["hera_mri"].append(float(mri(P_hera, args.top_k).item()))

            if args.compute_oracle_hit:
                oracle_ids = compute_oracle_expert_ids(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    P_ori=route_probs,
                    P_hera=P_hera,
                    target_layer=args.target_layer,
                    top_l=args.top_l_oracle,
                    bias_scale=args.lambda_r,
                ).cpu()
                oracle_logs["proto"].append(oracle_ids)
                oracle_logs["proto_health"].append(oracle_ids)
                oracle_logs["hera"].append(oracle_ids)

            state_bank.update_after_step(
                sample_repr=sample_repr.detach(),
                sample_prior=P_hera.detach(),
                risk=risk.detach(),
                low_threshold=args.tau_low,
                high_threshold=args.tau_high,
                eta_e=0.9,
                eta_q=0.95,
                eta_c=0.95,
            )
            global_batch_step += 1
            task_pbar.set_postfix_str(
                f"warmup={min(global_batch_step, args.warmup_batches)}/{args.warmup_batches} "
                f"logged={len(logs['task_name'])}"
            )
            task_pbar.update(1)
            overall_pbar.update(1)

        task_pbar.close()

    overall_pbar.close()

    if not logs["P_ori"]:
        raise RuntimeError(
            "No post-warmup routing logs were collected. Reduce warmup_batches or increase the number of examples."
        )

    domain_ids = torch.cat(logs["domain_id"], dim=0)
    sample_repr = torch.cat(logs["sample_repr"], dim=0)
    risk = torch.cat(logs["risk"], dim=0)
    P_ori = torch.cat(logs["P_ori"], dim=0)
    P_proto = torch.cat(logs["P_proto"], dim=0)
    P_proto_health = torch.cat(logs["P_proto_health"], dim=0)
    P_hera = torch.cat(logs["P_hera"], dim=0)

    oracle_proto = None
    oracle_proto_health = None
    oracle_hera = None
    if args.compute_oracle_hit and oracle_logs["hera"]:
        oracle_proto = torch.cat(oracle_logs["proto"], dim=0)
        oracle_proto_health = torch.cat(oracle_logs["proto_health"], dim=0)
        oracle_hera = torch.cat(oracle_logs["hera"], dim=0)

    num_domains = len(task_to_domain)
    metrics = {
        "Original Router": evaluate_routing_metrics(
            P=P_ori,
            sample_repr=sample_repr,
            domain_ids=domain_ids,
            num_domains=num_domains,
            risk=risk,
            oracle_expert_ids=oracle_hera if args.compute_oracle_hit else None,
            top_k=args.top_k,
        ),
        "HERA-Proto": evaluate_routing_metrics(
            P=P_proto,
            sample_repr=sample_repr,
            domain_ids=domain_ids,
            num_domains=num_domains,
            risk=risk,
            oracle_expert_ids=oracle_proto if args.compute_oracle_hit else None,
            top_k=args.top_k,
        ),
        "HERA-Proto+Health": evaluate_routing_metrics(
            P=P_proto_health,
            sample_repr=sample_repr,
            domain_ids=domain_ids,
            num_domains=num_domains,
            risk=risk,
            oracle_expert_ids=oracle_proto_health if args.compute_oracle_hit else None,
            top_k=args.top_k,
        ),
        "HERA-Full": evaluate_routing_metrics(
            P=P_hera,
            sample_repr=sample_repr,
            domain_ids=domain_ids,
            num_domains=num_domains,
            risk=risk,
            oracle_expert_ids=oracle_hera if args.compute_oracle_hit else None,
            top_k=args.top_k,
        ),
    }

    stability = {key: summarize_batch_metric(values) for key, values in batch_stats.items()}
    tensor_log_path = os.path.join(args.output_path, "routing_logs.pt")
    json_path = os.path.join(args.output_path, "routing_metrics.json")

    torch.save(
        {
            "task_names": logs["task_name"],
            "domain_ids": domain_ids,
            "sample_repr": sample_repr,
            "risk": risk,
            "P_ori": P_ori,
            "P_proto": P_proto,
            "P_proto_health": P_proto_health,
            "P_hera": P_hera,
            "oracle_expert_ids": oracle_hera if oracle_hera is not None else None,
        },
        tensor_log_path,
    )

    payload = {
        "config": args.__dict__,
        "task_to_domain": task_to_domain,
        "num_logged_samples": int(P_ori.size(0)),
        "metrics": metrics,
        "batch_stability": stability,
        "artifacts": {
            "tensor_logs": tensor_log_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n========== Routing Metrics ==========")
    for method_name, result in metrics.items():
        print(f"[{method_name}]")
        for key, value in result.items():
            print(f"  {key}: {value:.6f}")
    print("=====================================")
    print(f"Metrics JSON: {json_path}")
    print(f"Tensor Logs:  {tensor_log_path}")


if __name__ == "__main__":
    Fire(run_routing_eval)
