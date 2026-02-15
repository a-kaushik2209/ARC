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
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.config import FailureMode, PredictionConfig

@dataclass
class ModelOutput:
    failure_probs: torch.Tensor
    time_to_failure: torch.Tensor
    ttf_uncertainty: torch.Tensor
    embeddings: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "failure_probs": self.failure_probs.detach().cpu().numpy(),
            "time_to_failure": self.time_to_failure.detach().cpu().numpy(),
            "ttf_uncertainty": self.ttf_uncertainty.detach().cpu().numpy(),
            "embeddings": self.embeddings.detach().cpu().numpy(),
            "attention_weights": self.attention_weights.detach().cpu().numpy() if self.attention_weights is not None else None,
        }

class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out + residual

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        out, weights = self.attention(x, x, x, need_weights=return_weights)
        out = self.norm(out + residual)

        return out, weights if return_weights else None

class TrainingDynamicsPredictor(nn.Module):
    def __init__(self, config: Optional[PredictionConfig] = None, n_features: int = 64):
        super().__init__()

        self.config = config or PredictionConfig()
        self.n_features = n_features
        self.n_failure_modes = len(FailureMode)

        hidden = self.config.temporal_hidden_size

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
        )

        self.cnn_blocks = nn.ModuleList()
        in_ch = hidden
        for i, out_ch in enumerate(self.config.cnn_channels):
            self.cnn_blocks.append(
                TemporalBlock(
                    in_ch, out_ch,
                    kernel_size=self.config.cnn_kernel_size,
                    dilation=2 ** i,
                    dropout=self.config.dropout_rate,
                )
            )
            in_ch = out_ch

        cnn_out_dim = self.config.cnn_channels[-1] if self.config.cnn_channels else hidden

        self.gru = nn.GRU(
            cnn_out_dim,
            hidden,
            num_layers=self.config.num_gru_layers,
            bidirectional=True,
            dropout=self.config.dropout_rate if self.config.num_gru_layers > 1 else 0,
            batch_first=True,
        )

        gru_out_dim = hidden * 2

        self.attention = TemporalAttention(gru_out_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.failure_head = nn.Sequential(
            nn.Linear(gru_out_dim, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(hidden, self.n_failure_modes + 1),
        )

        self.ttf_head = nn.Sequential(
            nn.Linear(gru_out_dim, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(hidden, self.n_failure_modes * 2),
        )

        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> ModelOutput:
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)

        x = x.transpose(1, 2)
        for block in self.cnn_blocks:
            x = block(x)
        x = x.transpose(1, 2)

        x, _ = self.gru(x)

        x, attn_weights = self.attention(x, return_weights=return_attention)

        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)

        failure_logits = self.failure_head(pooled)
        failure_probs = F.softmax(failure_logits / self.temperature, dim=-1)

        ttf_out = self.ttf_head(pooled)
        ttf_mean = ttf_out[:, :self.n_failure_modes]
        ttf_log_var = ttf_out[:, self.n_failure_modes:]
        ttf_uncertainty = torch.exp(0.5 * ttf_log_var)

        ttf_mean = F.softplus(ttf_mean)

        return ModelOutput(
            failure_probs=failure_probs,
            time_to_failure=ttf_mean,
            ttf_uncertainty=ttf_uncertainty,
            embeddings=pooled,
            attention_weights=attn_weights,
        )

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        n_samples = n_samples or self.config.mc_dropout_samples

        self.train()

        failure_samples = []
        ttf_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                failure_samples.append(output.failure_probs)
                ttf_samples.append(output.time_to_failure)

        failure_stack = torch.stack(failure_samples, dim=0)
        ttf_stack = torch.stack(ttf_samples, dim=0)

        return {
            "failure_probs_mean": failure_stack.mean(dim=0),
            "failure_probs_std": failure_stack.std(dim=0),
            "ttf_mean": ttf_stack.mean(dim=0),
            "ttf_std": ttf_stack.std(dim=0),
        }

    def get_confidence_interval(
        self,
        probs_mean: torch.Tensor,
        probs_std: torch.Tensor,
        level: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = 1.96 if level == 0.95 else 2.576

        lower = torch.clamp(probs_mean - z * probs_std, 0, 1)
        upper = torch.clamp(probs_mean + z * probs_std, 0, 1)

        return lower, upper

    def get_feature_importance(
        self,
        x: torch.Tensor,
        target_mode: int,
    ) -> torch.Tensor:
        x = x.requires_grad_(True)

        output = self.forward(x)
        target_prob = output.failure_probs[:, target_mode]

        grad = torch.autograd.grad(
            target_prob.sum(),
            x,
            create_graph=False,
        )[0]

        importance = (x * grad).abs()

        return importance.detach()

class EnsemblePredictor(nn.Module):

    def __init__(
        self,
        n_models: int = 5,
        config: Optional[PredictionConfig] = None,
        n_features: int = 64,
    ):
        super().__init__()

        self.models = nn.ModuleList([
            TrainingDynamicsPredictor(config, n_features)
            for _ in range(n_models)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = [model(x) for model in self.models]

        probs = torch.stack([o.failure_probs for o in outputs], dim=0)
        ttf = torch.stack([o.time_to_failure for o in outputs], dim=0)

        return {
            "failure_probs_mean": probs.mean(dim=0),
            "failure_probs_std": probs.std(dim=0),
            "ttf_mean": ttf.mean(dim=0),
            "ttf_std": ttf.std(dim=0),
        }