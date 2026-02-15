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

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
import warnings

try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

@dataclass
class AMPConfig:
    dtype: str = "fp16"
    use_autocast: bool = True

    detect_overflow: bool = True
    overflow_threshold: float = 65504.0
    bf16_overflow_threshold: float = 3.4e38

    coordinate_scaler: bool = True
    min_scale: float = 1.0
    scale_growth_interval: int = 2000

    rollback_on_overflow: bool = True
    reduce_scale_on_rollback: bool = True
    scale_reduction_factor: float = 0.5

class AMPRollback:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional['GradScaler'] = None,
        config: Optional[AMPConfig] = None,
        verbose: bool = True,
    ):
        from arc.intervention.rollback import WeightRollback, RollbackConfig

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config or AMPConfig()
        self.verbose = verbose

        if self.config.dtype == "bf16":
            self._overflow_threshold = self.config.bf16_overflow_threshold
        else:
            self._overflow_threshold = self.config.overflow_threshold

        rollback_config = RollbackConfig(
            fast_grad_norm=True,
            skip_stable_layers=True,
        )
        self._rollback = WeightRollback(model, optimizer, rollback_config, verbose=False)

        self._scale_history = []
        self._overflow_count = 0

        if verbose:
            print(f"⚡ AMPRollback initialized (dtype={self.config.dtype})")

    def _detect_overflow(self, loss: torch.Tensor) -> Dict[str, Any]:
        result = {
            'overflow': False,
            'type': None,
            'details': {},
        }

        if isinstance(loss, torch.Tensor):
            loss_val = loss.item() if loss.numel() == 1 else loss.max().item()

            if loss_val > self._overflow_threshold:
                result['overflow'] = True
                result['type'] = 'loss_overflow'
                result['details']['loss'] = loss_val
                result['details']['threshold'] = self._overflow_threshold
                return result

        for name, p in self.model.named_parameters():
            if p.dtype in [torch.float16, torch.bfloat16]:
                if p.abs().max().item() > self._overflow_threshold * 0.9:
                    result['overflow'] = True
                    result['type'] = 'weight_near_overflow'
                    result['details']['layer'] = name
                    result['details']['max_val'] = p.abs().max().item()
                    return result

        for p in self.model.parameters():
            if p.grad is not None:
                grad_max = p.grad.abs().max().item()
                if grad_max > self._overflow_threshold * 0.9:
                    result['overflow'] = True
                    result['type'] = 'gradient_near_overflow'
                    result['details']['max_grad'] = grad_max
                    return result

        return result

    def _handle_scaler_overflow(self) -> bool:
        if self.scaler is None or not self.config.coordinate_scaler:
            return False

        scale = self.scaler.get_scale()
        self._scale_history.append(scale)

        if len(self._scale_history) >= 2:
            if self._scale_history[-1] < self._scale_history[-2]:
                self._overflow_count += 1
                if self.verbose:
                    print(f"  GradScaler overflow detected (scale: {scale:.2f})")
                return True

        if scale < self.config.min_scale:
            self.scaler.update(self.config.min_scale)
            if self.verbose:
                print(f"  Restored GradScaler to min_scale ({self.config.min_scale})")

        return False

    def step(
        self,
        loss: torch.Tensor,
        scaler: Optional['GradScaler'] = None,
    ) -> 'AMPRollbackAction':
        if scaler is not None:
            self.scaler = scaler

        overflow = self._detect_overflow(loss)
        scaler_overflow = self._handle_scaler_overflow()

        action = AMPRollbackAction(
            step=self._rollback.state.step_count,
            rolled_back=False,
            overflow_detected=overflow['overflow'] or scaler_overflow,
            overflow_type=overflow.get('type'),
        )

        if overflow['overflow'] and self.config.rollback_on_overflow:
            rollback_action = self._rollback.step(torch.tensor(float('inf')))
            action.rolled_back = rollback_action.rolled_back
            action.steps_back = rollback_action.steps_back

            if self.scaler is not None and self.config.reduce_scale_on_rollback:
                current_scale = self.scaler.get_scale()
                new_scale = current_scale * self.config.scale_reduction_factor
                self.scaler.update(max(new_scale, self.config.min_scale))
                action.new_scale = self.scaler.get_scale()

            if self.verbose:
                print(f"  AMP Rollback: {overflow.get('type', 'scaler_overflow')}")
        else:
            rollback_action = self._rollback.step(loss)
            action.rolled_back = rollback_action.rolled_back
            action.steps_back = rollback_action.steps_back

        if self.scaler is not None:
            action.current_scale = self.scaler.get_scale()

        return action

    def end_epoch(self):
        self._rollback.end_epoch()
        self._overflow_count = 0

    def get_stats(self) -> Dict[str, Any]:
        stats = self._rollback.get_stats()
        stats.update({
            'overflow_count': self._overflow_count,
            'current_scale': self.scaler.get_scale() if self.scaler else None,
            'dtype': self.config.dtype,
        })
        return stats

@dataclass
class AMPRollbackAction:
    step: int
    rolled_back: bool
    overflow_detected: bool
    overflow_type: Optional[str] = None
    steps_back: int = 0
    current_scale: Optional[float] = None
    new_scale: Optional[float] = None

class SafeAutocast:

    def __init__(
        self,
        amp_rollback: Optional[AMPRollback] = None,
        device_type: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        enabled: bool = True,
    ):
        self.amp_rollback = amp_rollback
        self.device_type = device_type
        self.enabled = enabled and AMP_AVAILABLE

        if dtype is None:
            if amp_rollback and amp_rollback.config.dtype == "bf16":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = dtype

    def __enter__(self):
        if self.enabled:
            self._autocast = torch.autocast(
                device_type=self.device_type,
                dtype=self.dtype,
            )
            return self._autocast.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled:
            return self._autocast.__exit__(*args)
        return False
def create_amp_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dtype: str = "fp16",
    device: str = "cuda",
) -> tuple:

    if not AMP_AVAILABLE:
        warnings.warn("AMP not available, using FP32")
        return model, None, None

    scaler = GradScaler()

    config = AMPConfig(dtype=dtype)

    amp_rollback = AMPRollback(model, optimizer, scaler, config)

    return model, scaler, amp_rollback

if __name__ == "__main__":
    print("Testing AMPRollback...")

    model = nn.Linear(10, 2).cuda() if torch.cuda.is_available() else nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if AMP_AVAILABLE and torch.cuda.is_available():
        model, scaler, rollback = create_amp_training(model, optimizer, "fp16")
        print(f"  AMPRollback created: {rollback.get_stats()}")
    else:
        print("  AMP or CUDA not available, skipping test")