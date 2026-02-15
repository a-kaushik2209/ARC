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
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

@dataclass
class SelfKnowledge:
    predicted_gradient_norm: float
    actual_gradient_norm: float
    prediction_error: float
    confidence: float
    surprise: float
    future_loss_estimate: float

class GradientPredictor(nn.Module):
    def __init__(
        self,
        history_size: int = 20,
        hidden_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()

        self.history_size = history_size
        self.n_heads = n_heads

        input_dim = 10

        self.temporal_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 5),
            )
            for _ in range(n_heads)
        ])

        self._history: deque = deque(maxlen=history_size)

    def record(
        self,
        loss: float,
        gradient_norm: float,
        learning_rate: float,
        weight_norm: float,
        batch_variance: float,
        epoch: int,
    ) -> None:
        if len(self._history) > 0:
            prev = self._history[-1]
            loss_delta = loss - prev[0]
            grad_delta = gradient_norm - prev[1]
        else:
            loss_delta = 0.0
            grad_delta = 0.0

        features = [
            loss,
            gradient_norm,
            learning_rate,
            weight_norm,
            batch_variance,
            loss_delta,
            grad_delta,
            math.log(epoch + 1),
            math.sin(epoch * 0.1),
            math.cos(epoch * 0.1),
        ]

        self._history.append(features)

    def predict_next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self._history) < 5:
            return torch.zeros(5), torch.ones(5)

        history = list(self._history)
        x = torch.tensor(history, dtype=torch.float32).unsqueeze(0)

        encoded, _ = self.temporal_encoder(x)
        final_state = encoded[:, -1, :]

        predictions = []
        for head in self.prediction_heads:
            pred = head(final_state)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        mean = predictions.mean(dim=0).squeeze(0)
        var = predictions.var(dim=0).squeeze(0)

        return mean, var

    def compute_surprise(self, actual_gradient_norm: float) -> float:
        mean, var = self.predict_next()

        predicted_norm = mean[2].item()
        predicted_std = math.sqrt(var[2].item() + 1e-6)

        surprise = abs(actual_gradient_norm - predicted_norm) / predicted_std

        return float(surprise)

class MetaCognition(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        history_size: int = 50,
        prediction_horizon: int = 5,
    ):
        super().__init__()

        self.model = model
        self.prediction_horizon = prediction_horizon

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.gradient_predictor = GradientPredictor(history_size=history_size)

        self.state_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.loss_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, prediction_horizon),
        )

        self._loss_history: deque = deque(maxlen=history_size)
        self._param_sketches: deque = deque(maxlen=history_size)
        self._gradient_sketches: deque = deque(maxlen=history_size)

        self._prediction_errors: deque = deque(maxlen=100)

    def update(
        self,
        loss: float,
        gradient_norm: float,
        learning_rate: float,
    ) -> SelfKnowledge:
        weight_norm = sum(
            p.data.norm().item() ** 2
            for p in self.model.parameters()
        ) ** 0.5

        self.gradient_predictor.record(
            loss=loss,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            weight_norm=weight_norm,
            batch_variance=0.0,
            epoch=len(self._loss_history),
        )
        pred_mean, pred_var = self.gradient_predictor.predict_next()

        predicted_grad_norm = pred_mean[2].item() if len(pred_mean) > 2 else gradient_norm

        surprise = self.gradient_predictor.compute_surprise(gradient_norm)

        if len(self._loss_history) > 0:
            prediction_error = surprise
        else:
            prediction_error = 0.0

        self._prediction_errors.append(prediction_error)

        if len(self._prediction_errors) > 5:
            avg_error = np.mean(list(self._prediction_errors)[-10:])
            confidence = 1.0 / (1.0 + avg_error)
        else:
            confidence = 0.5

        future_loss = self._predict_future_loss(loss, gradient_norm)

        self._loss_history.append(loss)

        return SelfKnowledge(
            predicted_gradient_norm=predicted_grad_norm,
            actual_gradient_norm=gradient_norm,
            prediction_error=prediction_error,
            confidence=confidence,
            surprise=surprise,
            future_loss_estimate=future_loss,
        )

    def _predict_future_loss(self, current_loss: float, gradient_norm: float) -> float:
        if len(self._loss_history) < 2:
            return current_loss

        recent_losses = list(self._loss_history)[-5:]
        if len(recent_losses) >= 2:
            loss_velocity = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
        else:
            loss_velocity = 0.0

        future_loss = current_loss + self.prediction_horizon * loss_velocity

        return max(0, future_loss)

    def compute_self_modeling_loss(
        self,
        actual_loss: float,
        actual_gradient_norm: float,
    ) -> torch.Tensor:
        pred_mean, pred_var = self.gradient_predictor.predict_next()

        grad_loss = F.mse_loss(
            pred_mean[2:3],
            torch.tensor([actual_gradient_norm]),
        )

        loss_loss = F.mse_loss(
            pred_mean[0:1],
            torch.tensor([actual_loss]),
        )

        return grad_loss + loss_loss

class SelfModelingHead(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.metacognition = MetaCognition(model)

        self._step = 0
        self._learning_rate = 0.001

    def set_learning_rate(self, lr: float) -> None:
        self._learning_rate = lr

    def introspect(self, loss: float) -> SelfKnowledge:
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        knowledge = self.metacognition.update(
            loss=loss,
            gradient_norm=grad_norm,
            learning_rate=self._learning_rate,
        )

        self._step += 1

        return knowledge

    def get_self_modeling_loss(self, loss: float) -> torch.Tensor:
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2

        return self.metacognition.compute_self_modeling_loss(loss, grad_norm ** 0.5)