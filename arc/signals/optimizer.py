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

from typing import Dict, Any, Optional, List
import torch
import numpy as np

from arc.signals.base import SignalCollector
from arc.config import SignalConfig

class OptimizerCollector(SignalCollector):

    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        self.config = config or SignalConfig()
        self._optimizer_type: Optional[str] = None
        self._param_groups_info: List[Dict] = []

    def _register_hooks(self) -> None:

        if self._optimizer is None:
            return

        opt_class = self._optimizer.__class__.__name__
        self._optimizer_type = opt_class

        for i, group in enumerate(self._optimizer.param_groups):
            self._param_groups_info.append({
                "index": i,
                "lr": group.get("lr", 0),
                "weight_decay": group.get("weight_decay", 0),
            })

    def _collect_signals(self) -> Dict[str, Any]:

        signals = {
            "optimizer_type": self._optimizer_type,
            "param_groups": [],
            "global": {},
        }

        if self._optimizer is None:
            return signals

        all_momentum_norms = []
        all_m_norms = []
        all_v_norms = []
        all_v_means = []

        for group_idx, group in enumerate(self._optimizer.param_groups):
            group_signals = {
                "lr": group.get("lr", 0),
                "weight_decay": group.get("weight_decay", 0),
            }

            for param in group["params"]:
                if param not in self._optimizer.state:
                    continue

                state = self._optimizer.state[param]

                if "momentum_buffer" in state:
                    momentum = state["momentum_buffer"]
                    if momentum is not None:
                        norm = torch.norm(momentum, p=2).item()
                        all_momentum_norms.append(norm)

                if "exp_avg" in state:
                    m = state["exp_avg"]
                    m_norm = torch.norm(m, p=2).item()
                    all_m_norms.append(m_norm)

                if "exp_avg_sq" in state:
                    v = state["exp_avg_sq"]
                    v_norm = torch.norm(v, p=2).item()
                    v_mean = v.mean().item()
                    all_v_norms.append(v_norm)
                    all_v_means.append(v_mean)

            if "betas" in group:
                group_signals["beta1"] = group["betas"][0]
                group_signals["beta2"] = group["betas"][1]

            if "eps" in group:
                group_signals["eps"] = group["eps"]

            signals["param_groups"].append(group_signals)

        if all_momentum_norms:
            signals["global"]["total_momentum_norm"] = np.sum(all_momentum_norms)
            signals["global"]["mean_momentum_norm"] = np.mean(all_momentum_norms)

        if all_m_norms:
            signals["global"]["total_adam_m_norm"] = np.sum(all_m_norms)
            signals["global"]["mean_adam_m_norm"] = np.mean(all_m_norms)

        if all_v_norms:
            signals["global"]["total_adam_v_norm"] = np.sum(all_v_norms)
            signals["global"]["mean_adam_v_norm"] = np.mean(all_v_norms)

        if all_v_means:
            signals["global"]["mean_adam_v_mean"] = np.mean(all_v_means)
            signals["global"]["max_adam_v_mean"] = np.max(all_v_means)

        step_count = self._get_step_count()
        if step_count is not None:
            signals["global"]["step_count"] = step_count

        if self._optimizer.param_groups:
            signals["global"]["effective_lr"] = self._optimizer.param_groups[0].get("lr", 0)

        if step_count and all_m_norms and "betas" in self._optimizer.param_groups[0]:
            betas = self._optimizer.param_groups[0]["betas"]
            beta1, beta2 = betas

            m_correction = 1 - beta1 ** step_count
            v_correction = 1 - beta2 ** step_count

            if m_correction > 0 and all_m_norms:
                signals["global"]["corrected_m_norm"] = np.mean(all_m_norms) / m_correction
            if v_correction > 0 and all_v_norms:
                signals["global"]["corrected_v_norm"] = np.mean(all_v_norms) / v_correction

        return signals

    def _get_step_count(self) -> Optional[int]:

        if self._optimizer is None:
            return None

        for group in self._optimizer.param_groups:
            for param in group["params"]:
                if param in self._optimizer.state:
                    state = self._optimizer.state[param]
                    if "step" in state:
                        step = state["step"]
                        if isinstance(step, torch.Tensor):
                            return step.item()
                        return step
        return None

    def _get_metadata(self) -> Dict[str, Any]:

        base = super()._get_metadata()
        base.update({
            "optimizer_type": self._optimizer_type,
            "n_param_groups": len(self._param_groups_info),
        })
        return base