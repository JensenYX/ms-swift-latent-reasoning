"""Query-anchored latent policy for Res-CoLaR experiments.

This module is intentionally separate from ``latent_policy.py`` so the current
CoLaR implementation remains unchanged.  The policy can condition the latent
head on a prompt/query hidden state and can inject a zero-initialized gated
query residual into the latent embedding that is fed back to the LLM.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class QueryAnchoredLatentPolicy(nn.Module):
    """Latent policy with optional query conditioning and residual anchoring.

    The distribution target stays in the original CoLaR normalized latent space:
    ``gold_embedding / embeds_std``.  The query residual is applied only to the
    embedding fed back into the text backbone:

        anchored = (base + alpha * anchor) / sqrt(1 + alpha^2)

    where ``base`` and ``anchor`` are normalized latent vectors.  Multiplication
    by ``embeds_std`` happens outside this module when building inputs_embeds.
    """

    def __init__(
        self,
        feature_size: int,
        intermediate_size: int = 512,
        deterministic: bool = False,
        condition_head: bool = True,
        condition_interaction: bool = True,
        condition_input_norm: bool = False,
        residual_anchor: bool = True,
        norm_preserve: bool = True,
        gate_init: float = 0.0,
        proj_init: str = 'identity',
    ):
        super().__init__()
        self.feature_size = feature_size
        self.deterministic = deterministic
        self.condition_head = condition_head
        self.condition_interaction = condition_interaction
        self.condition_input_norm = condition_input_norm
        self.residual_anchor = residual_anchor
        self.norm_preserve = norm_preserve

        if condition_head:
            in_features = feature_size * (3 if condition_interaction else 2)
        else:
            in_features = feature_size
        self.fc = nn.Sequential(
            nn.Linear(in_features, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )
        self.mean = nn.Linear(intermediate_size, feature_size)
        if not deterministic:
            self.log_std = nn.Linear(intermediate_size, feature_size)

        self.query_proj = nn.Linear(feature_size, feature_size)
        self.anchor_gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self._init_query_proj(proj_init)

    def _init_query_proj(self, proj_init: str) -> None:
        proj_init = (proj_init or 'default').lower()
        if proj_init == 'default':
            return
        if proj_init == 'identity':
            nn.init.eye_(self.query_proj.weight)
            nn.init.zeros_(self.query_proj.bias)
            return
        if proj_init == 'zero':
            nn.init.zeros_(self.query_proj.weight)
            nn.init.zeros_(self.query_proj.bias)
            return
        raise ValueError(f'Unknown query projection init: {proj_init!r}')

    def _expand_query(self, x: torch.Tensor, query_hidden: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if query_hidden is None:
            return None
        q = query_hidden.to(device=x.device, dtype=x.dtype)
        if q.dim() == 2:
            q = q.unsqueeze(1)
        if q.shape[1] == 1 and x.shape[1] != 1:
            q = q.expand(-1, x.shape[1], -1)
        return q

    def _condition_inputs(
        self,
        x: torch.Tensor,
        query_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = self._expand_query(x, query_hidden)
        if q is None:
            q = torch.zeros_like(x)
        if not self.condition_input_norm:
            parts = [x, q]
            if self.condition_interaction:
                parts.append(x * q)
            return torch.cat(parts, dim=-1)

        x_norm = F.layer_norm(x, (x.shape[-1],))
        q_norm = F.layer_norm(q, (q.shape[-1],))
        parts = [x_norm, q_norm]
        if self.condition_interaction:
            parts.append(F.layer_norm(x_norm * q_norm, (x_norm.shape[-1],)))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        query_hidden: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.distributions.Normal:
        if self.condition_head:
            x_in = self._condition_inputs(x, query_hidden)
        else:
            x_in = x

        h = self.fc(x_in)
        mean = self.mean(h)
        if self.deterministic:
            return torch.distributions.Normal(mean, torch.ones_like(mean) * 1e-9)
        log_std = self.log_std(h)
        std = log_std.exp() * temperature
        return torch.distributions.Normal(mean, std)

    def anchor_normalized(self, query_hidden: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        q = query_hidden.to(device=like.device, dtype=torch.float32)
        if q.dim() == 2:
            q = q.unsqueeze(1)
        anchor = self.query_proj(q)
        return _rms_normalize(anchor)

    def alpha(self) -> torch.Tensor:
        return self.anchor_gate.float()

    def apply_anchor_normalized(
        self,
        base_normalized: torch.Tensor,
        query_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (not self.residual_anchor) or query_hidden is None:
            return base_normalized
        anchor = self.anchor_normalized(query_hidden, base_normalized).to(
            device=base_normalized.device,
            dtype=base_normalized.dtype,
        )
        if anchor.shape[1] == 1 and base_normalized.shape[1] != 1:
            anchor = anchor.expand(-1, base_normalized.shape[1], -1)
        alpha = self.alpha().to(device=base_normalized.device, dtype=base_normalized.dtype)
        mixed = base_normalized + alpha * anchor
        if self.norm_preserve:
            mixed = mixed / torch.sqrt(1.0 + alpha.square())
        return mixed

    def apply_anchor_to_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        query_hidden: Optional[torch.Tensor],
        latent_input_mask: torch.Tensor,
        embeds_std: float,
    ) -> torch.Tensor:
        if (not self.residual_anchor) or query_hidden is None:
            return inputs_embeds
        if latent_input_mask.sum().item() <= 0:
            return inputs_embeds

        base_norm = inputs_embeds.float() / float(embeds_std)
        anchored_norm = self.apply_anchor_normalized(base_norm, query_hidden)
        anchored_embeds = anchored_norm * float(embeds_std)

        mask = latent_input_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).unsqueeze(-1)
        return inputs_embeds * (1.0 - mask) + anchored_embeds.to(inputs_embeds.dtype) * mask
