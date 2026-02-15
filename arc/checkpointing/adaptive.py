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
import os
import tempfile
import shutil
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import warnings
import gc
import struct
import io

class CheckpointStrategy(Enum):
    FULL_CPU = auto()
    QUANTIZED_FP16 = auto()
    QUANTIZED_INT8 = auto()
    INCREMENTAL_DELTA = auto()
    STREAMING_DISK = auto()
    SHARDED = auto()

@dataclass
class AdaptiveCheckpointConfig:
    auto_select_strategy: bool = True
    preferred_strategy: Optional[CheckpointStrategy] = None

    full_cpu_threshold: float = 4.0
    quantized_threshold: float = 2.0
    incremental_threshold: float = 1.0

    delta_threshold: float = 1e-6
    delta_check_layers: int = 10

    quantize_optimizer: bool = True

    disk_checkpoint_dir: str = "/tmp/arc_checkpoints"
    max_disk_checkpoints: int = 3
    compress_disk: bool = False

    shard_count: int = 4

    max_checkpoints: int = 3
    checkpoint_frequency: int = 50
    verbose: bool = True

@dataclass
class CheckpointMetadata:
    step: int
    strategy: CheckpointStrategy
    model_size_bytes: int
    checkpoint_size_bytes: int
    compression_ratio: float
    timestamp: float
    is_incremental: bool = False
    base_checkpoint_idx: Optional[int] = None

class AdaptiveCheckpointer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[AdaptiveCheckpointConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or AdaptiveCheckpointConfig()

        self.checkpoints: deque = deque(maxlen=self.config.max_checkpoints)
        self.metadata: List[CheckpointMetadata] = []

        self.step = 0
        self.current_strategy: Optional[CheckpointStrategy] = None
        self._last_full_state: Optional[Dict[str, torch.Tensor]] = None

        self.model_size_bytes = self._calculate_model_size()
        self.optimizer_size_bytes = self._calculate_optimizer_size()

        if self.config.disk_checkpoint_dir:
            os.makedirs(self.config.disk_checkpoint_dir, exist_ok=True)

        if self.config.auto_select_strategy:
            self.current_strategy = self._select_best_strategy()
        else:
            self.current_strategy = self.config.preferred_strategy or CheckpointStrategy.FULL_CPU

        if self.config.verbose:
            self._print_init_info()

    def _calculate_model_size(self) -> int:
        total = 0
        for param in self.model.parameters():
            total += param.numel() * param.element_size()
        for buffer in self.model.buffers():
            total += buffer.numel() * buffer.element_size()
        return total

    def _calculate_optimizer_size(self) -> int:
        total = 0
        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        return total

    def _get_available_memory(self) -> Dict[str, int]:
        import psutil

        memory = {
            "cpu_available": psutil.virtual_memory().available,
            "cpu_total": psutil.virtual_memory().total,
        }

        if torch.cuda.is_available():
            memory["gpu_available"] = torch.cuda.get_device_properties(0).total_memory -                                      torch.cuda.memory_allocated()
            memory["gpu_total"] = torch.cuda.get_device_properties(0).total_memory

        return memory

    def _select_best_strategy(self) -> CheckpointStrategy:
        try:
            memory = self._get_available_memory()
            cpu_available = memory["cpu_available"]

            total_size = self.model_size_bytes + self.optimizer_size_bytes

            if cpu_available > total_size * 3 * self.config.full_cpu_threshold:
                return CheckpointStrategy.FULL_CPU
            elif cpu_available > total_size * 1.5 * self.config.quantized_threshold:
                return CheckpointStrategy.QUANTIZED_FP16
            elif cpu_available > total_size * 0.3 * self.config.incremental_threshold:
                return CheckpointStrategy.INCREMENTAL_DELTA
            else:
                return CheckpointStrategy.STREAMING_DISK

        except Exception:
            return CheckpointStrategy.QUANTIZED_FP16

    def _print_init_info(self):
        print(f"AdaptiveCheckpointer initialized")
        print(f"   Model size: {self.model_size_bytes / 1e9:.2f} GB")
        print(f"   Optimizer size: {self.optimizer_size_bytes / 1e9:.2f} GB")
        print(f"   Strategy: {self.current_strategy.name}")

    def save(self, step: Optional[int] = None) -> CheckpointMetadata:
        self.step = step or self.step + 1

        if self.current_strategy == CheckpointStrategy.FULL_CPU:
            return self._save_full_cpu()
        elif self.current_strategy == CheckpointStrategy.QUANTIZED_FP16:
            return self._save_quantized(dtype=torch.float16)
        elif self.current_strategy == CheckpointStrategy.QUANTIZED_INT8:
            return self._save_quantized(dtype=torch.int8)
        elif self.current_strategy == CheckpointStrategy.INCREMENTAL_DELTA:
            return self._save_incremental()
        elif self.current_strategy == CheckpointStrategy.STREAMING_DISK:
            return self._save_to_disk()
        else:
            return self._save_full_cpu()

    def _save_full_cpu(self) -> CheckpointMetadata:
        import time
        start_time = time.time()

        checkpoint = {
            'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'optimizer': {
                k: {
                    k2: v2.cpu().clone() if isinstance(v2, torch.Tensor) else v2
                    for k2, v2 in v.items()
                }
                for k, v in self.optimizer.state.items()
            },
            'step': self.step,
        }

        checkpoint['rng'] = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        checkpoint_size = sum(
            v.numel() * v.element_size()
            for v in checkpoint['model'].values()
        )

        self.checkpoints.append(checkpoint)
        self._last_full_state = checkpoint['model']

        metadata = CheckpointMetadata(
            step=self.step,
            strategy=CheckpointStrategy.FULL_CPU,
            model_size_bytes=self.model_size_bytes,
            checkpoint_size_bytes=checkpoint_size,
            compression_ratio=1.0,
            timestamp=time.time(),
        )
        self.metadata.append(metadata)

        if self.config.verbose:
            print(f"   Saved full CPU checkpoint at step {self.step} ({checkpoint_size / 1e6:.1f} MB)")

        return metadata

    def _save_quantized(self, dtype: torch.dtype = torch.float16) -> CheckpointMetadata:
        import time

        def quantize_state_dict(state_dict: Dict) -> Dict:
            quantized = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    quantized[k] = v.cpu().to(dtype)
                elif isinstance(v, torch.Tensor):
                    quantized[k] = v.cpu().clone()
                else:
                    quantized[k] = v
            return quantized

        checkpoint = {
            'model': quantize_state_dict(self.model.state_dict()),
            'optimizer': {},
            'step': self.step,
            'dtype': str(dtype),
        }

        if self.config.quantize_optimizer:
            for k, v in self.optimizer.state.items():
                checkpoint['optimizer'][k] = {
                    k2: (v2.cpu().to(dtype) if isinstance(v2, torch.Tensor) and v2.is_floating_point()
                         else v2.cpu().clone() if isinstance(v2, torch.Tensor) else v2)
                    for k2, v2 in v.items()
                }

        checkpoint_size = sum(
            v.numel() * v.element_size()
            for v in checkpoint['model'].values()
        )

        self.checkpoints.append(checkpoint)

        compression = self.model_size_bytes / checkpoint_size if checkpoint_size > 0 else 1.0

        metadata = CheckpointMetadata(
            step=self.step,
            strategy=CheckpointStrategy.QUANTIZED_FP16 if dtype == torch.float16 else CheckpointStrategy.QUANTIZED_INT8,
            model_size_bytes=self.model_size_bytes,
            checkpoint_size_bytes=checkpoint_size,
            compression_ratio=compression,
            timestamp=time.time(),
        )
        self.metadata.append(metadata)

        if self.config.verbose:
            print(f"   Saved quantized checkpoint ({compression:.1f}x compression)")

        return metadata

    def _save_incremental(self) -> CheckpointMetadata:
        import time

        current_state = self.model.state_dict()

        if self._last_full_state is None:
            return self._save_full_cpu()

        delta = {}
        for name, param in current_state.items():
            if name in self._last_full_state:
                old_param = self._last_full_state[name]
                if not torch.allclose(param.cpu(), old_param, atol=self.config.delta_threshold):
                    delta[name] = param.cpu().clone()
            else:
                delta[name] = param.cpu().clone()

        checkpoint = {
            'delta': delta,
            'base_idx': len(self.checkpoints) - 1,
            'step': self.step,
            'is_incremental': True,
        }

        checkpoint_size = sum(
            v.numel() * v.element_size()
            for v in delta.values()
        )

        self.checkpoints.append(checkpoint)

        compression = self.model_size_bytes / checkpoint_size if checkpoint_size > 0 else float('inf')

        self._last_full_state = {k: v.cpu().clone() for k, v in current_state.items()}

        metadata = CheckpointMetadata(
            step=self.step,
            strategy=CheckpointStrategy.INCREMENTAL_DELTA,
            model_size_bytes=self.model_size_bytes,
            checkpoint_size_bytes=checkpoint_size,
            compression_ratio=compression,
            timestamp=time.time(),
            is_incremental=True,
            base_checkpoint_idx=len(self.checkpoints) - 2,
        )
        self.metadata.append(metadata)

        if self.config.verbose:
            print(f"   Saved incremental checkpoint ({len(delta)}/{len(current_state)} params changed, {compression:.1f}x compression)")

        return metadata

    def _save_to_disk(self) -> CheckpointMetadata:
        import time

        self._clean_disk_checkpoints()

        filename = os.path.join(
            self.config.disk_checkpoint_dir,
            f"checkpoint_{self.step}.pt"
        )

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
        }

        torch.save(checkpoint, filename)

        checkpoint_size = os.path.getsize(filename)

        self.checkpoints.append({'path': filename, 'step': self.step})

        metadata = CheckpointMetadata(
            step=self.step,
            strategy=CheckpointStrategy.STREAMING_DISK,
            model_size_bytes=self.model_size_bytes,
            checkpoint_size_bytes=checkpoint_size,
            compression_ratio=1.0,
            timestamp=time.time(),
        )
        self.metadata.append(metadata)

        if self.config.verbose:
            print(f"   Saved to disk: {filename}")

        return metadata

    def _clean_disk_checkpoints(self):
        files = sorted([
            os.path.join(self.config.disk_checkpoint_dir, f)
            for f in os.listdir(self.config.disk_checkpoint_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ], key=os.path.getmtime)

        while len(files) >= self.config.max_disk_checkpoints:
            try:
                os.remove(files.pop(0))
            except Exception:
                pass

    def restore(self, checkpoint_idx: int = -1) -> int:
        if not self.checkpoints:
            raise ValueError("No checkpoints available")

        checkpoint = self.checkpoints[checkpoint_idx]

        if isinstance(checkpoint, dict) and 'path' in checkpoint:
            checkpoint = torch.load(checkpoint['path'])

        if checkpoint.get('is_incremental', False):
            checkpoint = self._resolve_incremental(checkpoint)

        if 'model' in checkpoint:
            model_state = checkpoint['model']
            device = next(self.model.parameters()).device
            restored_state = {}
            for k, v in model_state.items():
                if isinstance(v, torch.Tensor):
                    restored_state[k] = v.to(device=device, dtype=torch.float32)
                else:
                    restored_state[k] = v
            self.model.load_state_dict(restored_state)

        if 'optimizer' in checkpoint and checkpoint['optimizer']:
            if isinstance(list(checkpoint['optimizer'].values())[0] if checkpoint['optimizer'] else None, dict):
                device = next(self.model.parameters()).device
                for k, v in checkpoint['optimizer'].items():
                    if k in self.optimizer.state:
                        for k2, v2 in v.items():
                            if isinstance(v2, torch.Tensor):
                                self.optimizer.state[k][k2] = v2.to(device=device, dtype=torch.float32)

        if 'rng' in checkpoint:
            torch.set_rng_state(checkpoint['rng']['torch'])
            if checkpoint['rng']['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])

        restored_step = checkpoint.get('step', 0)

        if self.config.verbose:
            print(f"   Restored checkpoint from step {restored_step}")

        return restored_step

    def _resolve_incremental(self, checkpoint: Dict) -> Dict:
        base_idx = checkpoint.get('base_idx', 0)
        base = self.checkpoints[base_idx]

        if base.get('is_incremental', False):
            base = self._resolve_incremental(base)

        full_state = base['model'].copy()
        for k, v in checkpoint['delta'].items():
            full_state[k] = v

        return {
            'model': full_state,
            'optimizer': base.get('optimizer', {}),
            'step': checkpoint['step'],
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.current_strategy.name,
            "num_checkpoints": len(self.checkpoints),
            "model_size_gb": self.model_size_bytes / 1e9,
            "optimizer_size_gb": self.optimizer_size_bytes / 1e9,
            "total_saved_checkpoints": len(self.metadata),
            "avg_compression_ratio": sum(m.compression_ratio for m in self.metadata) / len(self.metadata) if self.metadata else 0,
        }

def create_adaptive_checkpointer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    **kwargs
) -> AdaptiveCheckpointer:
    config = AdaptiveCheckpointConfig(**kwargs)
    return AdaptiveCheckpointer(model, optimizer, config)