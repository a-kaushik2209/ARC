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
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
import warnings

try:
    from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TrainerCallback = object

class HuggingFaceCallback(TrainerCallback if HF_AVAILABLE else object):
    def __init__(
        self,
        loss_threshold: float = 100.0,
        gradient_threshold: float = 1e4,
        lr_spike_factor: float = 100.0,
        enable_rollback: bool = True,
        verbose: bool = True,
    ):
        if not HF_AVAILABLE:
            raise ImportError("transformers package required for HuggingFaceCallback")

        super().__init__()

        self.loss_threshold = loss_threshold
        self.gradient_threshold = gradient_threshold
        self.lr_spike_factor = lr_spike_factor
        self.enable_rollback = enable_rollback
        self.verbose = verbose

        self._initial_lr = None
        self._last_loss = None
        self._rollback_count = 0
        self._last_good_step = 0
        self._loss_history = []

        self._rollback = None

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if model is not None and optimizer is not None and self.enable_rollback:
            from arc.intervention.rollback import WeightRollback, RollbackConfig

            config = RollbackConfig(
                loss_explosion_threshold=self.loss_threshold,
                gradient_explosion_threshold=self.gradient_threshold,
                lr_spike_factor=self.lr_spike_factor,
                checkpoint_frequency=50,
                fast_grad_norm=True,
            )
            self._rollback = WeightRollback(model, optimizer, config, verbose=False)

            for pg in optimizer.param_groups:
                self._initial_lr = pg['lr']
                break

            if self.verbose:
                print(f"  ARC HuggingFace Callback: Protection enabled (loss_threshold={self.loss_threshold})")

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self._rollback is None:
            return

        if state.log_history:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                loss = last_log['loss']
                self._loss_history.append(loss)

                loss_tensor = torch.tensor(loss)
                action = self._rollback.step(loss_tensor)

                if action.rolled_back:
                    self._rollback_count += 1
                    if self.verbose:
                        print(f"  ARC: Rollback #{self._rollback_count} at step {state.global_step} ({action.failure_type})")

                    control.should_training_stop = False
                else:
                    self._last_good_step = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._rollback is not None:
            self._rollback.end_epoch()

    def on_train_end(self, args, state, control, **kwargs):
        if self.verbose and self._rollback is not None:
            stats = self._rollback.get_stats()
            print(f"  ARC Summary: {stats['total_rollbacks']} rollbacks, {stats['checkpoints_saved']} checkpoints")

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    Callback = object

class LightningCallback(Callback if LIGHTNING_AVAILABLE else object):
    def __init__(
        self,
        loss_threshold: float = 100.0,
        gradient_threshold: float = 1e4,
        enable_rollback: bool = True,
        verbose: bool = True,
    ):
        if not LIGHTNING_AVAILABLE:
            raise ImportError("pytorch_lightning package required for LightningCallback")

        super().__init__()

        self.loss_threshold = loss_threshold
        self.gradient_threshold = gradient_threshold
        self.enable_rollback = enable_rollback
        self.verbose = verbose

        self._rollback = None
        self._rollback_count = 0

    def on_train_start(self, trainer, pl_module):
        if self.enable_rollback:
            from arc.intervention.rollback import WeightRollback, RollbackConfig

            config = RollbackConfig(
                loss_explosion_threshold=self.loss_threshold,
                gradient_explosion_threshold=self.gradient_threshold,
                checkpoint_frequency=50,
                fast_grad_norm=True,
            )

            optimizers = trainer.optimizers
            if optimizers:
                optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
                self._rollback = WeightRollback(pl_module, optimizer, config, verbose=False)

                if self.verbose:
                    print(f"  ARC Lightning Callback: Protection enabled")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._rollback is None:
            return

        loss = None
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        elif hasattr(outputs, 'loss'):
            loss = outputs.loss

        if loss is not None:
            if isinstance(loss, torch.Tensor):
                action = self._rollback.step(loss.detach())
            else:
                action = self._rollback.step(torch.tensor(loss))

            if action.rolled_back:
                self._rollback_count += 1
                if self.verbose:
                    print(f"  ARC: Rollback #{self._rollback_count} at batch {batch_idx}")

    def on_train_epoch_end(self, trainer, pl_module):
        if self._rollback is not None:
            self._rollback.end_epoch()

    def on_train_end(self, trainer, pl_module):
        if self.verbose and self._rollback is not None:
            stats = self._rollback.get_stats()
            print(f"  ARC Summary: {stats['total_rollbacks']} rollbacks")

class YOLOCallback:

    def __init__(
        self,
        loss_threshold: float = 100.0,
        box_loss_threshold: float = 20.0,
        cls_loss_threshold: float = 50.0,
        enable_rollback: bool = True,
        verbose: bool = True,
    ):
        self.loss_threshold = loss_threshold
        self.box_loss_threshold = box_loss_threshold
        self.cls_loss_threshold = cls_loss_threshold
        self.enable_rollback = enable_rollback
        self.verbose = verbose

        self._rollback = None
        self._rollback_count = 0
        self._initialized = False

        self._trainer = None
        self._model = None
        self._optimizer = None

    def _lazy_init(self, trainer):
        if self._initialized:
            return

        self._trainer = trainer
        self._model = trainer.model
        self._optimizer = trainer.optimizer

        if self.enable_rollback and self._model is not None and self._optimizer is not None:
            from arc.intervention.rollback import WeightRollback, RollbackConfig
            from arc.intervention.heuristics import DetectionHeuristics, DetectionConfig

            config = RollbackConfig(
                loss_explosion_threshold=self.loss_threshold,
                checkpoint_frequency=100,
                fast_grad_norm=True,
                adaptive_frequency=True,
            )
            self._rollback = WeightRollback(self._model, self._optimizer, config, verbose=False)

            det_config = DetectionConfig(
                box_loss_threshold=self.box_loss_threshold,
                cls_loss_threshold=self.cls_loss_threshold,
            )
            self._heuristics = DetectionHeuristics(self._model, det_config, verbose=False)

            if self.verbose:
                print(f"  ARC YOLO Callback: Protection enabled")

        self._initialized = True

    def on_train_batch_end(self, trainer):
        self._lazy_init(trainer)

        if self._rollback is None:
            return

        loss = getattr(trainer, 'loss', None)
        if loss is None and hasattr(trainer, 'tloss'):
            loss = trainer.tloss

        if loss is not None:
            if isinstance(loss, (list, tuple)):
                loss = sum(loss) if all(isinstance(l, torch.Tensor) for l in loss) else loss[0]

            if isinstance(loss, torch.Tensor):
                action = self._rollback.step(loss.detach())

                if action.rolled_back:
                    self._rollback_count += 1
                    if self.verbose:
                        print(f"  ARC: Rollback #{self._rollback_count} ({action.failure_type})")

    def on_train_epoch_end(self, trainer):
        if self._rollback is not None:
            self._rollback.end_epoch()

    def on_train_end(self, trainer):
        if self.verbose and self._rollback is not None:
            stats = self._rollback.get_stats()
            print(f"  ARC YOLO Summary: {stats['total_rollbacks']} rollbacks")

    def wrap_trainer(self, trainer):
        if hasattr(trainer, 'add_callback'):
            trainer.add_callback('on_train_batch_end', self.on_train_batch_end)
            trainer.add_callback('on_train_epoch_end', self.on_train_epoch_end)
            trainer.add_callback('on_train_end', self.on_train_end)
        else:
            original_batch_end = getattr(trainer, 'on_train_batch_end', None)

            def wrapped_batch_end(*args, **kwargs):
                if original_batch_end:
                    original_batch_end(*args, **kwargs)
                self.on_train_batch_end(trainer)

            trainer.on_train_batch_end = wrapped_batch_end

        if self.verbose:
            print(f"  ARC YOLO: Callbacks registered")

class GenericTrainingWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_threshold: float = 100.0,
        gradient_threshold: float = 1e4,
        enable_heuristics: bool = True,
        verbose: bool = True,
    ):
        from arc.intervention.rollback import WeightRollback, RollbackConfig

        config = RollbackConfig(
            loss_explosion_threshold=loss_threshold,
            gradient_explosion_threshold=gradient_threshold,
            fast_grad_norm=True,
            adaptive_frequency=True,
        )

        self._rollback = WeightRollback(model, optimizer, config, verbose=verbose)

        self._heuristics = None
        if enable_heuristics:
            try:
                from arc.intervention.heuristics import AutoHeuristics
                self._heuristics = AutoHeuristics(model, verbose=False)
            except Exception:
                pass

        self.verbose = verbose

    def step(self, loss: torch.Tensor, output: Optional[torch.Tensor] = None) -> 'WrapperAction':
        action = self._rollback.step(loss)

        result = WrapperAction(
            skip_step=action.rolled_back,
            rolled_back=action.rolled_back,
            failure_type=action.failure_type,
        )

        if self._heuristics is not None and output is not None:
            heuristic_result = self._heuristics.check(output)
            if heuristic_result['critical']:
                result.warnings.extend(heuristic_result['warnings'])

        return result

    def end_epoch(self):
        self._rollback.end_epoch()

    def get_stats(self) -> Dict[str, Any]:
        return self._rollback.get_stats()

@dataclass
class WrapperAction:
    skip_step: bool
    rolled_back: bool
    failure_type: Optional[str] = None
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

def get_callback(framework: str, **kwargs):
    framework = framework.lower()

    if framework in ["huggingface", "hf", "transformers"]:
        return HuggingFaceCallback(**kwargs)
    elif framework in ["lightning", "pl", "pytorch_lightning"]:
        return LightningCallback(**kwargs)
    elif framework in ["yolo", "ultralytics"]:
        return YOLOCallback(**kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}. Use 'huggingface', 'lightning', or 'yolo'")

if __name__ == "__main__":
    print("Testing Framework Integrations...")

    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())

    wrapper = GenericTrainingWrapper(model, optimizer, verbose=True)

    for i in range(10):
        x = torch.randn(4, 10)
        out = model(x)
        loss = out.mean()
        loss.backward()

        action = wrapper.step(loss, out)
        if action.skip_step:
            print(f"Step {i}: Skipped due to rollback")
            optimizer.zero_grad()
            continue

        optimizer.step()
        optimizer.zero_grad()

    print(f"  GenericTrainingWrapper test complete: {wrapper.get_stats()}")

    print(f"\n  Available integrations:")
    print(f"  - HuggingFace: {HF_AVAILABLE}")
    print(f"  - PyTorch Lightning: {LIGHTNING_AVAILABLE}")
    print(f"  - Ultralytics YOLO: Always available (lazy import)")