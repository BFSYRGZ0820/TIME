from typing import List

import torch


def _get_nested_attr(root, path: List):
    obj = root
    for p in path:
        if isinstance(p, int):
            obj = obj[p]
        else:
            obj = getattr(obj, p)
    return obj


def resolve_moe_gate_module(model, layer_idx: int):
    candidates = [
        ["base_model", "model", "model", "layers", layer_idx, "mlp", "gate"],
        ["model", "model", "layers", layer_idx, "mlp", "gate"],
        ["model", "layers", layer_idx, "mlp", "gate"],
    ]
    last_error = None
    for path in candidates:
        try:
            return _get_nested_attr(model, path)
        except Exception as exc:  # pragma: no cover - fallback path probing
            last_error = exc
    raise ValueError(f"Cannot resolve gate module for layer {layer_idx}: {last_error}")


def calibrate_router_logits(
    router_logits: torch.Tensor,
    sample_prior: torch.Tensor,
    lam: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    prior_log = torch.log(sample_prior + eps).unsqueeze(1)
    return router_logits + lam * prior_log


class HeraRouterCalibrator:
    def __init__(self, model, layer_idx: int, lam: float = 2.0, eps: float = 1e-6):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.lam = float(lam)
        self.eps = float(eps)

    def attach(self, sample_prior: torch.Tensor, batch_size: int):
        gate = resolve_moe_gate_module(self.model, self.layer_idx)
        prior = sample_prior.detach()

        def hook(_module, _inputs, outputs):
            logits = outputs
            if logits.dim() == 2:
                total_token, num_experts = logits.shape
                seq_len = max(1, total_token // max(1, batch_size))
                logits_view = logits.view(batch_size, seq_len, num_experts)
                logits_new = calibrate_router_logits(logits_view, prior.to(logits.device), self.lam, self.eps)
                return logits_new.view_as(logits)

            if logits.dim() == 3:
                logits_new = calibrate_router_logits(logits, prior.to(logits.device), self.lam, self.eps)
                return logits_new

            return logits

        return gate.register_forward_hook(hook)
