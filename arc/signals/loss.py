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
from collections import deque
import numpy as np

from arc.signals.base import SignalCollector
from arc.config import SignalConfig

class LossCollector(SignalCollector):

    def __init__(self, config: Optional[SignalConfig] = None, ema_alpha: float = 0.1):
        super().__init__(config)
        self.config = config or SignalConfig()
        self.ema_alpha = ema_alpha

        self._batch_losses: deque = deque(maxlen=100)

        self._epoch_train_losses: List[float] = []
        self._epoch_val_losses: List[float] = []
        self._train_loss_ema: Optional[float] = None

        self._initial_loss: Optional[float] = None
        self._best_val_loss: float = float('inf')
        self._epochs_since_improvement: int = 0
        self._current_train_loss: Optional[float] = None
        self._current_val_loss: Optional[float] = None

    def _register_hooks(self) -> None:

        pass

    def record_batch_loss(self, loss: float) -> None:

        if np.isnan(loss) or np.isinf(loss):

            self._batch_losses.append(float('nan') if np.isnan(loss) else float('inf'))
        else:
            self._batch_losses.append(loss)

        if self._train_loss_ema is None:
            self._train_loss_ema = loss
        elif not np.isnan(loss) and not np.isinf(loss):
            self._train_loss_ema = (
                self.ema_alpha * loss +
                (1 - self.ema_alpha) * self._train_loss_ema
            )

    def record_epoch_loss(
        self,
        train_loss: float,
        val_loss: Optional[float] = None
    ) -> None:

        self._current_train_loss = train_loss
        self._epoch_train_losses.append(train_loss)

        if self._initial_loss is None:
            self._initial_loss = train_loss

        if val_loss is not None:
            self._current_val_loss = val_loss
            self._epoch_val_losses.append(val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._epochs_since_improvement = 0
            else:
                self._epochs_since_improvement += 1

    def _collect_signals(self) -> Dict[str, Any]:

        signals = {
            "batch": {},
            "epoch": {},
            "trajectory": {},
        }

        if self._batch_losses:
            batch_array = np.array(list(self._batch_losses))
            valid_losses = batch_array[np.isfinite(batch_array)]

            signals["batch"]["n_batches"] = len(self._batch_losses)
            signals["batch"]["n_nan"] = np.isnan(batch_array).sum()
            signals["batch"]["n_inf"] = np.isinf(batch_array).sum()

            if len(valid_losses) > 0:
                signals["batch"]["mean"] = valid_losses.mean()
                signals["batch"]["std"] = valid_losses.std() if len(valid_losses) > 1 else 0.0
                signals["batch"]["variance"] = valid_losses.var() if len(valid_losses) > 1 else 0.0
                signals["batch"]["min"] = valid_losses.min()
                signals["batch"]["max"] = valid_losses.max()

        if self._current_train_loss is not None:
            signals["epoch"]["train_loss"] = self._current_train_loss

        if self._train_loss_ema is not None:
            signals["epoch"]["train_loss_ema"] = self._train_loss_ema

        if self._current_val_loss is not None:
            signals["epoch"]["val_loss"] = self._current_val_loss

        if self._current_train_loss is not None and self._current_val_loss is not None:
            if self._current_val_loss > 1e-8:
                gap = (self._current_val_loss - self._current_train_loss) / self._current_val_loss
                signals["epoch"]["train_val_gap"] = gap

            if self._current_train_loss > 1e-8:
                ratio = self._current_val_loss / self._current_train_loss
                signals["epoch"]["val_train_ratio"] = ratio

        signals["epoch"]["best_val_loss"] = self._best_val_loss if self._best_val_loss < float('inf') else None
        signals["epoch"]["epochs_since_improvement"] = self._epochs_since_improvement

        signals["trajectory"]["initial_loss"] = self._initial_loss

        if self._initial_loss is not None and self._current_train_loss is not None:
            if self._initial_loss > 1e-8:
                signals["trajectory"]["loss_ratio"] = self._current_train_loss / self._initial_loss

        if len(self._epoch_train_losses) >= 2:
            gradient = self._epoch_train_losses[-1] - self._epoch_train_losses[-2]
            signals["trajectory"]["loss_gradient"] = gradient

            if len(self._epoch_train_losses) >= 3:
                prev_gradient = self._epoch_train_losses[-2] - self._epoch_train_losses[-3]
                acceleration = gradient - prev_gradient
                signals["trajectory"]["loss_acceleration"] = acceleration

        if len(self._epoch_train_losses) >= 5:
            recent = self._epoch_train_losses[-5:]
            x = np.arange(5)

            slope = np.polyfit(x, recent, 1)[0]
            signals["trajectory"]["loss_trend"] = slope

        self._batch_losses.clear()

        return signals

    def reset(self) -> None:

        super().reset()
        self._batch_losses.clear()
        self._epoch_train_losses.clear()
        self._epoch_val_losses.clear()
        self._train_loss_ema = None
        self._initial_loss = None
        self._best_val_loss = float('inf')
        self._epochs_since_improvement = 0
        self._current_train_loss = None
        self._current_val_loss = None

    def _get_metadata(self) -> Dict[str, Any]:

        base = super()._get_metadata()
        base.update({
            "n_epochs_recorded": len(self._epoch_train_losses),
            "has_val_loss": len(self._epoch_val_losses) > 0,
            "ema_alpha": self.ema_alpha,
        })
        return base