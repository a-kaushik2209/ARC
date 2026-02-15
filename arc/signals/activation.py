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
import random
import torch
import torch.nn as nn
import numpy as np

from arc.signals.base import SignalCollector
from arc.config import SignalConfig

class WelfordAccumulator:

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float) -> None:

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def update_batch(self, values: np.ndarray) -> None:

        for v in values.flat:
            self.update(float(v))

    @property
    def variance(self) -> float:

        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:

        return np.sqrt(self.variance)

    def reset(self) -> None:

        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

class ActivationCollector(SignalCollector):

    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        self.config = config or SignalConfig()

        self._activation_stats: Dict[str, WelfordAccumulator] = {}
        self._sparsity_accum: Dict[str, List[float]] = {}
        self._last_activations: Dict[str, torch.Tensor] = {}
        self._dead_neuron_counts: Dict[str, np.ndarray] = {}
        self._total_samples: Dict[str, int] = {}
        self._layer_names: List[str] = []

        self._prev_activations: Dict[str, torch.Tensor] = {}

    def _register_hooks(self) -> None:

        if self._model is None:
            return

        target_layers = self._get_target_layers()

        for name, module in self._model.named_modules():

            if target_layers and name not in target_layers:
                continue

            if self._is_activation_layer(module):
                self._layer_names.append(name or "root")

                hook = module.register_forward_hook(
                    self._make_forward_hook(name or "root")
                )
                self._hooks.append(hook)

    def _get_target_layers(self) -> Optional[List[str]]:

        layers = self.config.activation_sample_layers
        if layers:
            return layers
        return None

    def _is_activation_layer(self, module: nn.Module) -> bool:

        activation_types = (
            nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.GELU,
            nn.Tanh, nn.Sigmoid, nn.Softmax, nn.LogSoftmax,
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.LSTM, nn.GRU, nn.TransformerEncoderLayer,
        )
        return isinstance(module, activation_types)

    def _make_forward_hook(self, layer_name: str):

        def hook(module, input, output):

            if random.random() > self.config.activation_sample_ratio:
                return

            if isinstance(output, tuple):
                output = output[0]

            if not isinstance(output, torch.Tensor):
                return

            act = output.detach()
            self._process_activations(layer_name, act, module)

        return hook

    def _process_activations(
        self,
        layer_name: str,
        activations: torch.Tensor,
        module: nn.Module
    ) -> None:

        if activations.dim() > 2:
            act_flat = activations.flatten(start_dim=2).mean(dim=2)
        else:
            act_flat = activations

        if layer_name not in self._activation_stats:
            self._activation_stats[layer_name] = WelfordAccumulator()
            self._sparsity_accum[layer_name] = []
            self._total_samples[layer_name] = 0

            n_features = act_flat.shape[-1] if act_flat.dim() > 1 else act_flat.numel()
            self._dead_neuron_counts[layer_name] = np.zeros(n_features)

        mean_val = act_flat.mean().item()
        self._activation_stats[layer_name].update(mean_val)

        sparsity = (act_flat.abs() < 1e-6).float().mean().item()
        self._sparsity_accum[layer_name].append(sparsity)

        if act_flat.dim() > 1:
            active_mask = (act_flat.abs() > 1e-6).any(dim=0).cpu().numpy()
            self._dead_neuron_counts[layer_name] += active_mask.astype(float)

        self._total_samples[layer_name] += 1

        self._last_activations[layer_name] = act_flat.mean(dim=0).cpu()

    def _collect_signals(self) -> Dict[str, Any]:

        signals = {
            "layer_stats": {},
            "global": {},
        }

        all_means = []
        all_stds = []
        all_sparsities = []
        all_dead_ratios = []
        all_similarities = []

        for layer_name in self._layer_names:
            stats = {}

            if layer_name in self._activation_stats:
                accum = self._activation_stats[layer_name]
                stats["activation_mean"] = accum.mean
                stats["activation_std"] = accum.std
                all_means.append(accum.mean)
                all_stds.append(accum.std)

            if layer_name in self._sparsity_accum and self._sparsity_accum[layer_name]:
                sparsity = np.mean(self._sparsity_accum[layer_name])
                stats["activation_sparsity"] = sparsity
                all_sparsities.append(sparsity)

            if layer_name in self._dead_neuron_counts:
                counts = self._dead_neuron_counts[layer_name]
                total = self._total_samples.get(layer_name, 1)
                dead_ratio = (counts == 0).sum() / max(len(counts), 1)
                stats["dead_neuron_ratio"] = dead_ratio
                all_dead_ratios.append(dead_ratio)

            if layer_name in self._last_activations and layer_name in self._prev_activations:
                current = self._last_activations[layer_name]
                previous = self._prev_activations[layer_name]

                if current.shape == previous.shape and current.numel() > 0:
                    similarity = self._cosine_similarity(current, previous)
                    stats["activation_similarity"] = similarity
                    all_similarities.append(similarity)

            signals["layer_stats"][layer_name] = stats

        if all_means:
            signals["global"]["mean_activation_mean"] = np.mean(all_means)
            signals["global"]["mean_activation_std"] = np.mean(all_stds)
        if all_sparsities:
            signals["global"]["mean_sparsity"] = np.mean(all_sparsities)
        if all_dead_ratios:
            signals["global"]["mean_dead_ratio"] = np.mean(all_dead_ratios)
        if all_similarities:
            signals["global"]["mean_similarity"] = np.mean(all_similarities)
            signals["global"]["max_similarity"] = np.max(all_similarities)

        self._prev_activations = {k: v.clone() for k, v in self._last_activations.items()}

        self._sparsity_accum = {k: [] for k in self._layer_names}

        return signals

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:

        a_flat = a.flatten().float()
        b_flat = b.flatten().float()

        a_norm = torch.norm(a_flat)
        b_norm = torch.norm(b_flat)

        if a_norm < 1e-8 or b_norm < 1e-8:
            return 1.0 if a_norm < 1e-8 and b_norm < 1e-8 else 0.0

        return (torch.dot(a_flat, b_flat) / (a_norm * b_norm)).item()

    def reset(self) -> None:

        super().reset()
        for accum in self._activation_stats.values():
            accum.reset()
        self._sparsity_accum = {k: [] for k in self._layer_names}
        self._dead_neuron_counts = {k: np.zeros_like(v) for k, v in self._dead_neuron_counts.items()}
        self._total_samples = {k: 0 for k in self._total_samples}
        self._last_activations.clear()
        self._prev_activations.clear()

    def _get_metadata(self) -> Dict[str, Any]:

        base = super()._get_metadata()
        base.update({
            "n_layers": len(self._layer_names),
            "layer_names": self._layer_names,
            "sample_ratio": self.config.activation_sample_ratio,
        })
        return base