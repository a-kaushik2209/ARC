# ARC (Automatic Recovery Controller) - Self-Healing Neural Networks
# Copyright (c) 2026 Aryan Kaushik. All rights reserved.
#
# This file is part of ARC.
#
# ARC is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# ARC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ARC. If not, see <https://www.gnu.org/licenses/>.

from typing import Dict, Any, Optional, List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        bias: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        x = F.silu(x)

        y = self._ssm(x)

        z = F.silu(z)
        output = y * z

        output = self.out_proj(output)

        return output

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_inner = x.shape

        A = -torch.exp(self.A_log)

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )

        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        dA = torch.exp(dt.unsqueeze(-1) * A)

        dB = dt.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)

        y = y + x * self.D

        return y

class MambaBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return x + residual

class MultiScaleTemporalFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        scales: List[int] = [5, 20, 50],
        num_heads: int = 4,
    ):
        super().__init__()

        self.scales = scales
        self.d_model = d_model

        self.scale_processors = nn.ModuleList([
            MambaBlock(d_model, d_state=8)
            for _ in scales
        ])

        self.scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1)
            for _ in scales
        ])

        self.fusion_attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        self.out_proj = nn.Linear(d_model * len(scales), d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape

        scale_outputs = []

        for i, (scale, processor, pool) in enumerate(
            zip(self.scales, self.scale_processors, self.scale_pools)
        ):
            window_size = min(scale, seq_len)
            x_scale = x[:, -window_size:, :]

            processed = processor(x_scale)

            pooled = pool(processed.transpose(1, 2)).squeeze(-1)
            scale_outputs.append(pooled)

        scale_stack = torch.stack(scale_outputs, dim=1)

        fused, _ = self.fusion_attention(scale_stack, scale_stack, scale_stack)

        fused_flat = fused.reshape(batch, -1)
        output = self.out_proj(fused_flat)

        return output, scale_stack

class SignalCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_signal_types: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_signal_types = num_signal_types

        self.signal_embeddings = nn.Embedding(num_signal_types, d_model)

        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        signal_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, n_signals, _ = x.shape

        if signal_type_ids is not None:
            type_emb = self.signal_embeddings(signal_type_ids)
            x = x + type_emb

        x_norm = self.norm1(x)
        attn_out, attn_weights = self.cross_attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        x = x + self.ff(self.norm2(x))

        return x, attn_weights

class MambaPredictor(nn.Module):
    def __init__(
        self,
        n_features: int = 64,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        n_failure_modes: int = 6,
        dropout: float = 0.1,
        temporal_scales: List[int] = [5, 20, 50],
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.n_failure_modes = n_failure_modes

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.multiscale = MultiScaleTemporalFusion(
            d_model=d_model,
            scales=temporal_scales,
        )

        self.signal_attention = SignalCrossAttention(
            d_model=d_model,
            num_signal_types=6,
        )

        self.evidence_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_failure_modes + 1),
            nn.Softplus(),
        )

        self.ttf_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_failure_modes * 2),
        )

        self.embedding_proj = nn.Linear(d_model, d_model // 2)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len, _ = x.shape

        x = self.input_proj(x)

        for layer in self.mamba_layers:
            x = layer(x)

        fused, scale_reps = self.multiscale(x)

        attended, attn_weights = self.signal_attention(scale_reps)
        final_rep = fused + attended.mean(dim=1)

        evidence = self.evidence_head(final_rep)

        alpha = evidence + 1
        probs = alpha / alpha.sum(dim=-1, keepdim=True)

        total_evidence = alpha.sum(dim=-1, keepdim=True)
        uncertainty = (self.n_failure_modes + 1) / total_evidence

        ttf_out = self.ttf_head(final_rep)
        ttf_mean = F.softplus(ttf_out[:, :self.n_failure_modes])
        ttf_log_var = ttf_out[:, self.n_failure_modes:]
        ttf_std = torch.exp(0.5 * ttf_log_var)

        embedding = self.embedding_proj(final_rep)

        output = {
            "failure_probs": probs,
            "evidence": evidence,
            "alpha": alpha,
            "uncertainty": uncertainty,
            "time_to_failure": ttf_mean,
            "ttf_std": ttf_std,
            "embedding": embedding,
        }

        if return_attention:
            output["attention_weights"] = attn_weights

        return output

    def evidential_loss(
        self,
        evidence: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
        n_epochs: int = 100,
    ) -> torch.Tensor:
        alpha = evidence + 1
        S = alpha.sum(dim=-1, keepdim=True)

        n_classes = evidence.size(-1)
        y_onehot = F.one_hot(targets, n_classes).float()

        loss_ml = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

        annealing = min(1.0, epoch / (n_epochs * 0.5))

        alpha_tilde = alpha - y_onehot * (alpha - 1)
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(n_classes, dtype=S.dtype, device=S.device))
            - torch.lgamma(alpha_tilde).sum(dim=-1)
            + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1)
        )

        loss = (loss_ml + annealing * kl).mean()

        return loss