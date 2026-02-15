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

from typing import Any, Optional
import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    Callback = object

from arc.config import Config
from arc.api.callback import Arc
from arc.prediction.predictor import FailurePrediction

class ArcCallback(Callback if HAS_LIGHTNING else object):
    def __init__(
        self,
        config: Optional[Config] = None,
        model_path: Optional[str] = None,
        auto_intervene: bool = False,
        verbose: bool = True,
        log_to_logger: bool = True,
    ):
        if not HAS_LIGHTNING:
            raise ImportError(
                "PyTorch Lightning not installed. "
                "Install with: pip install pytorch-lightning"
            )

        super().__init__()

        self.config = config
        self.model_path = model_path
        self.auto_intervene = auto_intervene
        self.verbose = verbose
        self.log_to_logger = log_to_logger

        self.prophet: Optional[Arc] = None
        self.last_prediction: Optional[FailurePrediction] = None

    def on_fit_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule'
    ) -> None:
        self.prophet = Arc(
            config=self.config,
            model_path=self.model_path,
            auto_intervene=self.auto_intervene,
            verbose=self.verbose,
        )

        optimizers = trainer.optimizers
        optimizer = optimizers[0] if optimizers else None

        self.prophet.attach(pl_module, optimizer)

    def on_fit_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule'
    ) -> None:
        if self.prophet is not None:
            self.prophet.detach()

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.prophet is None:
            return

        loss = self._extract_loss(outputs)
        if loss is not None:
            self.prophet.on_batch_end(loss, trainer.global_step)

    def on_train_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ) -> None:
        if self.prophet is None:
            return

        val_loss = None
        if 'val_loss' in trainer.callback_metrics:
            val_loss = trainer.callback_metrics['val_loss'].item()

        train_loss = None
        if 'train_loss' in trainer.callback_metrics:
            train_loss = trainer.callback_metrics['train_loss'].item()

        prediction = self.prophet.on_epoch_end(
            trainer.current_epoch,
            val_loss=val_loss,
            train_loss=train_loss,
        )

        self.last_prediction = prediction

        if self.log_to_logger and trainer.logger is not None:
            self._log_prediction(trainer, prediction)

    def _extract_loss(self, outputs: Any) -> Optional[float]:
        if outputs is None:
            return None

        if isinstance(outputs, dict):
            if 'loss' in outputs:
                loss = outputs['loss']
            elif 'minimize' in outputs:
                loss = outputs['minimize']
            else:
                return None
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            return None

        if isinstance(loss, torch.Tensor):
            return loss.detach().item()

        return float(loss)

    def _log_prediction(
        self,
        trainer: 'pl.Trainer',
        prediction: FailurePrediction
    ) -> None:
        for mode, prob in prediction.failure_probabilities.items():
            trainer.logger.log_metrics({
                f"arc/{mode.name.lower()}_prob": prob
            }, step=trainer.current_epoch)

        trainer.logger.log_metrics({
            "arc/overall_risk": prediction.overall_risk
        }, step=trainer.current_epoch)

        if self.prophet is not None:
            trainer.logger.log_metrics({
                "arc/overhead_pct": self.prophet.overhead_percentage
            }, step=trainer.current_epoch)

    def state_dict(self) -> dict:
        if self.prophet is None:
            return {}

        return {
            "buffer": self.prophet.predictor.buffer.to_json(),
            "normalizer": self.prophet.predictor.normalizer.to_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if self.prophet is None or not state_dict:
            return

        from arc.features.buffer import SignalBuffer
        from arc.features.normalizer import OnlineNormalizer

        if "buffer" in state_dict:
            self.prophet.predictor.buffer = SignalBuffer.from_json(state_dict["buffer"])

        if "normalizer" in state_dict:
            self.prophet.predictor.normalizer = OnlineNormalizer.from_dict(state_dict["normalizer"])

class ArcWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[Config] = None,
        **kwargs: Any,
    ):
        self.prophet = Arc(config=config, **kwargs)
        self.prophet.attach(model, optimizer)

    def on_batch_end(self, loss: float, step: Optional[int] = None) -> None:
        self.prophet.on_batch_end(loss, step)

    def on_epoch_end(
        self,
        epoch: int,
        val_loss: Optional[float] = None
    ) -> FailurePrediction:
        return self.prophet.on_epoch_end(epoch, val_loss)

    def __enter__(self) -> 'ArcWrapper':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.prophet.detach()
        return False