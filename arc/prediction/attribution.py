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

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn

@dataclass
class SignalAttribution:
    signal_name: str
    importance: float
    temporal_importance: List[float]
    direction: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "importance": self.importance,
            "temporal_importance": self.temporal_importance,
            "direction": self.direction,
        }

class AttributionEngine:

    def __init__(self, model: nn.Module):

        self.model = model

    def gradient_attribution(
        self,
        features: torch.Tensor,
        target_class: int,
        feature_names: Optional[List[str]] = None,
    ) -> List[SignalAttribution]:

        features = features.requires_grad_(True)

        output = self.model(features)
        target_prob = output.failure_probs[:, target_class]

        grad = torch.autograd.grad(
            target_prob.sum(),
            features,
            create_graph=False,
        )[0]

        importance = (grad * features).abs()

        importance = importance.mean(dim=0)

        feature_importance = importance.mean(dim=0)
        temporal_importance = importance.sum(dim=1)

        feature_importance = feature_importance.detach().cpu().numpy()
        if feature_importance.max() > 0:
            feature_importance = feature_importance / feature_importance.max()

        temporal_importance = temporal_importance.detach().cpu().numpy()

        grad_sign = grad.mean(dim=(0, 1)).sign().detach().cpu().numpy()

        attributions = []
        n_features = len(feature_importance)

        for i in range(n_features):
            name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"

            attributions.append(SignalAttribution(
                signal_name=name,
                importance=float(feature_importance[i]),
                temporal_importance=importance[:, i].detach().cpu().numpy().tolist(),
                direction=int(grad_sign[i]) if i < len(grad_sign) else 0,
            ))

        attributions.sort(key=lambda x: x.importance, reverse=True)

        return attributions

    def integrated_gradients(
        self,
        features: torch.Tensor,
        target_class: int,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None,
        feature_names: Optional[List[str]] = None,
    ) -> List[SignalAttribution]:

        if baseline is None:
            baseline = torch.zeros_like(features)

        scaled_inputs = [
            baseline + (float(i) / n_steps) * (features - baseline)
            for i in range(n_steps + 1)
        ]

        gradients = []
        for scaled in scaled_inputs:
            scaled = scaled.requires_grad_(True)
            output = self.model(scaled)
            target_prob = output.failure_probs[:, target_class]

            grad = torch.autograd.grad(
                target_prob.sum(),
                scaled,
                create_graph=False,
            )[0]

            gradients.append(grad)

        avg_gradients = torch.stack(gradients).mean(dim=0)

        integrated = (features - baseline) * avg_gradients

        importance = integrated.abs().mean(dim=(0, 1))

        importance = importance.detach().cpu().numpy()
        if importance.max() > 0:
            importance = importance / importance.max()

        direction = integrated.mean(dim=(0, 1)).sign().detach().cpu().numpy()

        attributions = []
        n_features = len(importance)

        for i in range(n_features):
            name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"

            temporal = integrated[0, :, i].abs().detach().cpu().numpy().tolist()

            attributions.append(SignalAttribution(
                signal_name=name,
                importance=float(importance[i]),
                temporal_importance=temporal,
                direction=int(direction[i]) if i < len(direction) else 0,
            ))

        attributions.sort(key=lambda x: x.importance, reverse=True)

        return attributions

    def attention_attribution(
        self,
        features: torch.Tensor,
    ) -> Dict[str, np.ndarray]:

        with torch.no_grad():
            output = self.model(features, return_attention=True)

        if output.attention_weights is None:
            return {"error": "Model does not expose attention weights"}

        weights = output.attention_weights.detach().cpu().numpy()

        return {
            "attention_weights": weights,
            "mean_attention": weights.mean(axis=1) if weights.ndim > 2 else weights,
        }

    def ablation_study(
        self,
        features: torch.Tensor,
        target_class: int,
        feature_names: Optional[List[str]] = None,
    ) -> List[SignalAttribution]:

        with torch.no_grad():

            base_output = self.model(features)
            base_prob = base_output.failure_probs[:, target_class].mean().item()

        n_features = features.shape[-1]
        importance = []
        directions = []

        for i in range(n_features):

            ablated = features.clone()
            ablated[:, :, i] = 0.0

            with torch.no_grad():
                ablated_output = self.model(ablated)
                ablated_prob = ablated_output.failure_probs[:, target_class].mean().item()

            delta = base_prob - ablated_prob
            importance.append(abs(delta))
            directions.append(1 if delta > 0 else -1)

        importance = np.array(importance)
        if importance.max() > 0:
            importance = importance / importance.max()

        attributions = []
        for i in range(n_features):
            name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"

            attributions.append(SignalAttribution(
                signal_name=name,
                importance=float(importance[i]),
                temporal_importance=[float(importance[i])] * features.shape[1],
                direction=directions[i],
            ))

        attributions.sort(key=lambda x: x.importance, reverse=True)

        return attributions

    def explain_prediction(
        self,
        features: torch.Tensor,
        target_class: int,
        method: str = "integrated_gradients",
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        if method == "integrated_gradients":
            attributions = self.integrated_gradients(
                features, target_class, feature_names=feature_names
            )
        elif method == "gradient":
            attributions = self.gradient_attribution(
                features, target_class, feature_names=feature_names
            )
        elif method == "ablation":
            attributions = self.ablation_study(
                features, target_class, feature_names=feature_names
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        attention = self.attention_attribution(features)

        top_attributions = attributions[:top_k]

        total_positive = sum(
            a.importance for a in attributions if a.direction > 0
        )
        total_negative = sum(
            a.importance for a in attributions if a.direction < 0
        )

        return {
            "target_class": target_class,
            "method": method,
            "top_contributors": [a.to_dict() for a in top_attributions],
            "all_attributions": [a.to_dict() for a in attributions],
            "attention": attention if "error" not in attention else None,
            "summary": {
                "n_positive_contributors": sum(1 for a in attributions if a.direction > 0),
                "n_negative_contributors": sum(1 for a in attributions if a.direction < 0),
                "total_positive_importance": total_positive,
                "total_negative_importance": total_negative,
            },
        }
