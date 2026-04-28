from typing import Dict

import torch


class RunningNorm:
    def __init__(self, momentum: float = 0.05, eps: float = 1e-6):
        self.momentum = momentum
        self.eps = eps
        self.ready = False
        self.mean = 0.0
        self.var = 1.0

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_detached = x.detach()
        batch_mean = x_detached.mean().item()
        batch_var = x_detached.var(unbiased=False).item()

        if not self.ready:
            self.mean = batch_mean
            self.var = max(batch_var, self.eps)
            self.ready = True
        else:
            m = self.momentum
            self.mean = (1.0 - m) * self.mean + m * batch_mean
            self.var = (1.0 - m) * self.var + m * max(batch_var, self.eps)

        denom = (self.var + self.eps) ** 0.5
        return (x - self.mean) / denom


def predictive_entropy_from_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    token_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    mask = attention_mask.float()
    return torch.sum(token_entropy * mask, dim=-1) / (torch.sum(mask, dim=-1) + 1e-10)


def routing_entropy_from_sample_probs(sample_route_probs: torch.Tensor) -> torch.Tensor:
    p = sample_route_probs
    return -torch.sum(p * torch.log(p + 1e-12), dim=-1)


def approximate_expert_disagreement(
    route_probs: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    top2_prob = torch.topk(route_probs, k=min(2, route_probs.size(-1)), dim=-1).values
    if top2_prob.size(-1) == 1:
        margin = top2_prob[..., 0]
    else:
        margin = top2_prob[..., 0] - top2_prob[..., 1]
    disagree_token = 1.0 - torch.clamp(margin, min=0.0, max=1.0)
    mask = attention_mask.float()
    return torch.sum(disagree_token * mask, dim=-1) / (torch.sum(mask, dim=-1) + 1e-10)


def prototype_mismatch(sample_embs: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    dists = torch.cdist(sample_embs, prototypes, p=2.0)
    return torch.min(dists, dim=-1).values ** 2


class HeraRiskEstimator:
    def __init__(self, weights=None):
        if weights is None:
            weights = (1.0, 0.5, 0.75, 1.0)
        self.w1, self.w2, self.w3, self.w4 = [float(v) for v in weights]
        self.stats: Dict[str, RunningNorm] = {
            "pred": RunningNorm(),
            "route": RunningNorm(),
            "dis": RunningNorm(),
            "mismatch": RunningNorm(),
        }

    def __call__(
        self,
        pred_entropy: torch.Tensor,
        route_entropy: torch.Tensor,
        disagreement: torch.Tensor,
        mismatch: torch.Tensor,
    ) -> torch.Tensor:
        z1 = self.stats["pred"].normalize(pred_entropy)
        z2 = self.stats["route"].normalize(route_entropy)
        z3 = self.stats["dis"].normalize(disagreement)
        z4 = self.stats["mismatch"].normalize(mismatch)
        score = self.w1 * z1 + self.w2 * z2 + self.w3 * z3 + self.w4 * z4
        return torch.sigmoid(score)
