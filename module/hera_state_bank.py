from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class HeraStateConfig:
    warmup_batches: int = 10
    eta_e: float = 0.9
    eta_q: float = 0.95
    eta_c: float = 0.95
    gamma_min: float = 0.2
    gamma_max: float = 0.8
    gamma_delta: float = 0.1
    eps: float = 1e-8


class ExpertStateBank:
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
        config: Optional[HeraStateConfig] = None,
    ):
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self.config = config or HeraStateConfig()

        self.anchor = torch.zeros(self.num_experts, self.hidden_dim, device=self.device)
        self.proto = torch.zeros(self.num_experts, self.hidden_dim, device=self.device)
        self.reliability = torch.full((self.num_experts,), 0.5, device=self.device)
        self.contamination = torch.zeros(self.num_experts, device=self.device)
        self.initialized = torch.zeros(self.num_experts, dtype=torch.bool, device=self.device)

        self._warmup_sum = torch.zeros_like(self.anchor)
        self._warmup_count = torch.zeros(self.num_experts, device=self.device)
        self._warmup_seen_batches = 0
        self.warmup_done = False

        self._prev_mu = None
        self._prev_rho = None
        self._prev_entropy = None
        self.last_delta = 0.0

    @property
    def warmup_batches(self) -> int:
        return self.config.warmup_batches

    def update_domain_state(
        self,
        batch_mu: torch.Tensor,
        batch_rho: torch.Tensor,
        batch_pred_entropy: torch.Tensor,
        lambda_rho: float = 1.0,
        lambda_h: float = 1.0,
    ) -> float:
        mu = batch_mu.detach()
        rho = batch_rho.detach()
        ent = batch_pred_entropy.detach().float()

        if self._prev_mu is None:
            delta = 0.0
        else:
            delta_mu = torch.norm(mu - self._prev_mu, p=2).item()
            delta_rho = torch.norm(rho - self._prev_rho, p=1).item()
            delta_h = torch.abs(ent - self._prev_entropy).item()
            delta = delta_mu + lambda_rho * delta_rho + lambda_h * delta_h

        self._prev_mu = mu.clone()
        self._prev_rho = rho.clone()
        self._prev_entropy = ent.clone()
        self.last_delta = float(delta)
        return self.last_delta

    def _mix_gamma(self) -> float:
        cfg = self.config
        ratio = self.last_delta / (self.last_delta + cfg.gamma_delta + cfg.eps)
        ratio = max(cfg.gamma_min, min(cfg.gamma_max, ratio))
        return float(ratio)

    def get_fused_prototypes(self) -> torch.Tensor:
        gamma = self._mix_gamma()
        fused = gamma * self.anchor + (1.0 - gamma) * self.proto
        return F.normalize(fused, p=2, dim=-1)

    def warmup_update(
        self,
        sample_embs: torch.Tensor,
        sample_route_probs: torch.Tensor,
        sample_risk: torch.Tensor,
        tau_low: float,
    ) -> None:
        top2 = torch.topk(sample_route_probs, k=min(2, self.num_experts), dim=-1).indices
        low_mask = sample_risk < tau_low
        for i in range(sample_embs.size(0)):
            if not bool(low_mask[i]):
                continue
            emb = sample_embs[i]
            for expert_idx in top2[i].tolist():
                self._warmup_sum[expert_idx] += emb
                self._warmup_count[expert_idx] += 1.0
        self._warmup_seen_batches += 1

    def finalize_warmup(self, global_mean: torch.Tensor) -> None:
        if self.warmup_done:
            return
        g = global_mean.detach()
        for j in range(self.num_experts):
            if self._warmup_count[j] > 0:
                self.anchor[j] = self._warmup_sum[j] / (self._warmup_count[j] + self.config.eps)
                self.initialized[j] = True
            else:
                self.anchor[j] = g
                self.initialized[j] = True
        self.anchor = F.normalize(self.anchor, p=2, dim=-1)
        self.proto = self.anchor.clone()
        self.warmup_done = True

    def compute_domain_prior(self, batch_mu: torch.Tensor, temperature: float) -> torch.Tensor:
        proto = self.get_fused_prototypes()
        mu = F.normalize(batch_mu.unsqueeze(0), p=2, dim=-1)
        sim = F.cosine_similarity(mu, proto, dim=-1) / max(temperature, self.config.eps)
        return torch.softmax(sim, dim=-1)

    def compute_instance_prior(self, sample_embs: torch.Tensor, temperature: float) -> torch.Tensor:
        proto = self.get_fused_prototypes()
        u = F.normalize(sample_embs, p=2, dim=-1)
        sim = torch.matmul(u, proto.transpose(0, 1)) / max(temperature, self.config.eps)
        return torch.softmax(sim, dim=-1)

    def compute_health_prior(self, lambda_q: float, lambda_c: float) -> torch.Tensor:
        score = lambda_q * self.reliability - lambda_c * self.contamination
        return torch.softmax(score, dim=-1)

    def combine_priors(
        self,
        domain_prior: torch.Tensor,
        instance_prior: torch.Tensor,
        health_prior: torch.Tensor,
        alpha_d: float,
        alpha_i: float,
        alpha_h: float,
    ) -> torch.Tensor:
        eps = self.config.eps
        log_prior = (
            alpha_d * torch.log(domain_prior + eps).unsqueeze(0)
            + alpha_i * torch.log(instance_prior + eps)
            + alpha_h * torch.log(health_prior + eps).unsqueeze(0)
        )
        return torch.softmax(log_prior, dim=-1)

    def safe_expert_indices(self, top_b: int) -> torch.Tensor:
        b = max(1, min(int(top_b), self.num_experts))
        health = self.reliability - self.contamination
        return torch.topk(health, k=b, dim=-1).indices

    def apply_safe_mask(
        self,
        sample_prior: torch.Tensor,
        risk: torch.Tensor,
        tau_high: float,
        top_b: int,
    ) -> torch.Tensor:
        high_mask = risk >= tau_high
        if high_mask.sum().item() == 0:
            return sample_prior

        safe_idx = self.safe_expert_indices(top_b)
        mask = torch.zeros(self.num_experts, dtype=torch.bool, device=sample_prior.device)
        mask[safe_idx] = True

        out = sample_prior.clone()
        out[high_mask][:, ~mask] = 0.0
        out_sum = out.sum(dim=-1, keepdim=True)
        out = out / (out_sum + self.config.eps)
        return out

    def compute_novelty(
        self,
        sample_embs: torch.Tensor,
        sample_prior: torch.Tensor,
        risk: torch.Tensor,
        tau_low: float,
    ) -> torch.Tensor:
        top2 = torch.topk(sample_prior, k=min(2, self.num_experts), dim=-1).indices
        low_mask = risk < tau_low

        accum = torch.zeros_like(self.proto)
        count = torch.zeros(self.num_experts, device=self.device)
        for i in range(sample_embs.size(0)):
            if not bool(low_mask[i]):
                continue
            for j in top2[i].tolist():
                accum[j] += sample_embs[i]
                count[j] += 1.0

        novelty = torch.zeros(self.num_experts, device=self.device)
        for j in range(self.num_experts):
            if count[j] > 0:
                mean_emb = accum[j] / (count[j] + self.config.eps)
                novelty[j] = torch.norm(mean_emb - self.proto[j], p=2)
        return novelty

    def compute_lr_multiplier(
        self,
        novelty: torch.Tensor,
        a_n: float,
        a_q: float,
        a_c: float,
        m_min: float,
        m_max: float,
    ) -> torch.Tensor:
        nov_scale = novelty / (novelty.mean() + self.config.eps)
        mult = a_n * nov_scale + a_q * (1.0 - self.reliability) + a_c * (1.0 - self.contamination)
        return torch.clamp(mult, min=m_min, max=m_max)

    def update_after_step(
        self,
        sample_embs: torch.Tensor,
        sample_prior: torch.Tensor,
        risk: torch.Tensor,
        tau_low: float,
        tau_high: float,
    ) -> None:
        top2 = torch.topk(sample_prior, k=min(2, self.num_experts), dim=-1).indices
        cfg = self.config

        low_accum = torch.zeros_like(self.proto)
        low_count = torch.zeros(self.num_experts, device=self.device)
        low_risk_sum = torch.zeros(self.num_experts, device=self.device)
        high_count = torch.zeros(self.num_experts, device=self.device)

        for i in range(sample_embs.size(0)):
            is_low = bool(risk[i] < tau_low)
            is_high = bool(risk[i] >= tau_high)
            for j in top2[i].tolist():
                if is_low:
                    low_accum[j] += sample_embs[i]
                    low_count[j] += 1.0
                    low_risk_sum[j] += risk[i]
                if is_high:
                    high_count[j] += 1.0

        for j in range(self.num_experts):
            low_n = low_count[j]
            high_n = high_count[j]
            total_n = low_n + high_n

            if low_n > 0:
                low_mean = low_accum[j] / (low_n + cfg.eps)
                low_mean = F.normalize(low_mean.unsqueeze(0), p=2, dim=-1).squeeze(0)
                self.proto[j] = cfg.eta_e * self.proto[j] + (1.0 - cfg.eta_e) * low_mean
                rel_target = 1.0 - (low_risk_sum[j] / (low_n + cfg.eps))
                self.reliability[j] = cfg.eta_q * self.reliability[j] + (1.0 - cfg.eta_q) * rel_target

            if total_n > 0:
                cont_target = high_n / (total_n + cfg.eps)
                self.contamination[j] = cfg.eta_c * self.contamination[j] + (1.0 - cfg.eta_c) * cont_target

        self.proto = F.normalize(self.proto, p=2, dim=-1)
        self.reliability = torch.clamp(self.reliability, 0.0, 1.0)
        self.contamination = torch.clamp(self.contamination, 0.0, 1.0)
