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
import sys
import time
import traceback
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import warnings
import functools

class HardwareErrorType(Enum):
    CUDA_ERROR = auto()
    GPU_MEMORY_CORRUPTION = auto()
    GPU_UNAVAILABLE = auto()
    DISK_FULL = auto()
    NETWORK_FAILURE = auto()
    DDP_COMMUNICATION_FAILURE = auto()
    DRIVER_ERROR = auto()
    DEVICE_TIMEOUT = auto()

@dataclass
class HardwareConfig:
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True

    allow_cpu_fallback: bool = True
    allow_device_switch: bool = True

    checkpoint_to_disk_on_oom: bool = True
    checkpoint_dir: str = "/tmp/arc_checkpoints"
    remote_checkpoint_url: Optional[str] = None

    ddp_timeout_seconds: float = 300.0
    isolate_failed_ranks: bool = True

    monitor_gpu_health: bool = True
    health_check_interval: int = 100

    verbose: bool = True

@dataclass
class HardwareRecoveryResult:
    success: bool
    error_type: Optional[HardwareErrorType] = None
    recovery_action: str = ""
    new_device: Optional[torch.device] = None
    details: Dict[str, Any] = field(default_factory=dict)

class HardwareRecoveryHandler:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[HardwareConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or HardwareConfig()

        self.original_device = next(model.parameters()).device
        self.current_device = self.original_device
        self.fallback_device = torch.device("cpu")

        self.error_count: Dict[HardwareErrorType, int] = {e: 0 for e in HardwareErrorType}
        self.recovery_count = 0
        self.step = 0

        self._last_health_check = 0
        self._gpu_healthy = True

        if self.config.checkpoint_to_disk_on_oom:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        if self.config.verbose:
            print(f" HardwareRecoveryHandler initialized")
            print(f"   Primary device: {self.original_device}")
            print(f"   Fallback device: {self.fallback_device}")

    def classify_error(self, error: Exception) -> Optional[HardwareErrorType]:
        error_str = str(error).lower()
        error_type_str = type(error).__name__.lower()

        if 'cuda' in error_str or 'cuda' in error_type_str:
            if 'out of memory' in error_str:
                return None
            if 'driver' in error_str:
                return HardwareErrorType.DRIVER_ERROR
            if 'device' in error_str and ('unavailable' in error_str or 'not found' in error_str):
                return HardwareErrorType.GPU_UNAVAILABLE
            if 'illegal memory access' in error_str or 'corruption' in error_str:
                return HardwareErrorType.GPU_MEMORY_CORRUPTION
            return HardwareErrorType.CUDA_ERROR

        if 'no space' in error_str or 'disk full' in error_str or 'enospc' in error_str:
            return HardwareErrorType.DISK_FULL

        if 'network' in error_str or 'connection' in error_str or 'timeout' in error_str:
            if 'nccl' in error_str or 'gloo' in error_str:
                return HardwareErrorType.DDP_COMMUNICATION_FAILURE
            return HardwareErrorType.NETWORK_FAILURE

        if 'nccl' in error_str or 'collective' in error_str:
            return HardwareErrorType.DDP_COMMUNICATION_FAILURE

        return None

    def try_recover(self, error: Exception) -> HardwareRecoveryResult:
        error_type = self.classify_error(error)

        if error_type is None:
            return HardwareRecoveryResult(
                success=False,
                details={"reason": "Not a hardware error", "original_error": str(error)}
            )

        self.error_count[error_type] += 1

        if self.config.verbose:
            print(f"Hardware error detected: {error_type.name}")

        if error_type == HardwareErrorType.CUDA_ERROR:
            return self._recover_cuda_error()
        elif error_type == HardwareErrorType.GPU_MEMORY_CORRUPTION:
            return self._recover_memory_corruption()
        elif error_type == HardwareErrorType.GPU_UNAVAILABLE:
            return self._recover_gpu_unavailable()
        elif error_type == HardwareErrorType.DISK_FULL:
            return self._recover_disk_full()
        elif error_type == HardwareErrorType.NETWORK_FAILURE:
            return self._recover_network_failure()
        elif error_type == HardwareErrorType.DDP_COMMUNICATION_FAILURE:
            return self._recover_ddp_failure()
        elif error_type == HardwareErrorType.DRIVER_ERROR:
            return self._recover_driver_error()

        return HardwareRecoveryResult(
            success=False,
            error_type=error_type,
            details={"reason": "No recovery handler for this error type"}
        )

    def _recover_cuda_error(self) -> HardwareRecoveryResult:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.error_count[HardwareErrorType.CUDA_ERROR] >= self.config.max_retries:
                if self.config.allow_cpu_fallback:
                    return self._fallback_to_cpu()

            self.recovery_count += 1
            return HardwareRecoveryResult(
                success=True,
                error_type=HardwareErrorType.CUDA_ERROR,
                recovery_action="Reset CUDA context",
            )
        except Exception as e:
            return self._fallback_to_cpu() if self.config.allow_cpu_fallback else                   HardwareRecoveryResult(success=False, details={"error": str(e)})

    def _recover_memory_corruption(self) -> HardwareRecoveryResult:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                self.model.to(self.current_device)

            self.recovery_count += 1
            return HardwareRecoveryResult(
                success=True,
                error_type=HardwareErrorType.GPU_MEMORY_CORRUPTION,
                recovery_action="Cleared GPU memory and reinitialized",
            )
        except Exception:
            return self._fallback_to_cpu()

    def _recover_gpu_unavailable(self) -> HardwareRecoveryResult:
        if self.config.allow_device_switch and torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                if device != self.current_device:
                    try:
                        self.model.to(device)
                        self.current_device = device
                        self.recovery_count += 1

                        if self.config.verbose:
                            print(f"   Switched to GPU {i}")

                        return HardwareRecoveryResult(
                            success=True,
                            error_type=HardwareErrorType.GPU_UNAVAILABLE,
                            recovery_action=f"Switched to GPU {i}",
                            new_device=device,
                        )
                    except Exception:
                        continue

        return self._fallback_to_cpu()

    def _recover_disk_full(self) -> HardwareRecoveryResult:
        try:
            if os.path.exists(self.config.checkpoint_dir):
                files = sorted([
                    os.path.join(self.config.checkpoint_dir, f)
                    for f in os.listdir(self.config.checkpoint_dir)
                    if f.endswith('.pt') or f.endswith('.pth')
                ], key=os.path.getmtime)

                removed = 0
                for f in files[:-2]:
                    try:
                        os.remove(f)
                        removed += 1
                    except Exception:
                        pass

                if removed > 0:
                    self.recovery_count += 1
                    return HardwareRecoveryResult(
                        success=True,
                        error_type=HardwareErrorType.DISK_FULL,
                        recovery_action=f"Removed {removed} old checkpoints",
                        details={"removed_count": removed}
                    )
        except Exception as e:
            pass

        if self.config.remote_checkpoint_url:
            return HardwareRecoveryResult(
                success=True,
                error_type=HardwareErrorType.DISK_FULL,
                recovery_action="Switching to remote checkpoint storage",
                details={"remote_url": self.config.remote_checkpoint_url}
            )

        return HardwareRecoveryResult(
            success=False,
            error_type=HardwareErrorType.DISK_FULL,
            details={"reason": "Could not free disk space"}
        )

    def _recover_network_failure(self) -> HardwareRecoveryResult:
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries):
            time.sleep(delay)

            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier(timeout=torch.distributed.timedelta(seconds=10))

                self.recovery_count += 1
                return HardwareRecoveryResult(
                    success=True,
                    error_type=HardwareErrorType.NETWORK_FAILURE,
                    recovery_action=f"Network recovered after {attempt + 1} retries",
                )
            except Exception:
                if self.config.exponential_backoff:
                    delay *= 2

        return HardwareRecoveryResult(
            success=False,
            error_type=HardwareErrorType.NETWORK_FAILURE,
            details={"reason": "Network did not recover after max retries"}
        )

    def _recover_ddp_failure(self) -> HardwareRecoveryResult:
        try:
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier(
                        timeout=torch.distributed.timedelta(seconds=self.config.ddp_timeout_seconds)
                    )
                except Exception:
                    if self.config.isolate_failed_ranks:
                        warnings.warn("DDP ranks out of sync, continuing with available ranks")

                self.recovery_count += 1
                return HardwareRecoveryResult(
                    success=True,
                    error_type=HardwareErrorType.DDP_COMMUNICATION_FAILURE,
                    recovery_action="DDP barrier recovered",
                )
        except Exception as e:
            pass

        return self._fallback_to_single_device()

    def _recover_driver_error(self) -> HardwareRecoveryResult:
        return self._fallback_to_cpu()

    def _fallback_to_cpu(self) -> HardwareRecoveryResult:
        if not self.config.allow_cpu_fallback:
            return HardwareRecoveryResult(
                success=False,
                details={"reason": "CPU fallback not allowed"}
            )

        try:
            self.model.cpu()
            self.current_device = torch.device("cpu")

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

            self.recovery_count += 1

            if self.config.verbose:
                print("   Fell back to CPU execution")

            return HardwareRecoveryResult(
                success=True,
                recovery_action="Fell back to CPU",
                new_device=torch.device("cpu"),
            )
        except Exception as e:
            return HardwareRecoveryResult(
                success=False,
                details={"reason": f"CPU fallback failed: {e}"}
            )

    def _fallback_to_single_device(self) -> HardwareRecoveryResult:
        try:
            if hasattr(self.model, 'module'):
                self.model = self.model.module

            self.recovery_count += 1

            if self.config.verbose:
                print("   Fell back to single device (DDP disabled)")

            return HardwareRecoveryResult(
                success=True,
                recovery_action="Disabled DDP, continuing on single device",
            )
        except Exception as e:
            return HardwareRecoveryResult(
                success=False,
                details={"reason": f"Single device fallback failed: {e}"}
            )

    def check_gpu_health(self) -> bool:
        if not torch.cuda.is_available():
            return True

        try:
            torch.cuda.synchronize()

            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            if free_memory < 100 * 1024 * 1024:
                warnings.warn("GPU memory critically low")
                return False

            return True
        except Exception:
            return False

    def safe_context(self):
        return SafeHardwareContext(self)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_device": str(self.current_device),
            "original_device": str(self.original_device),
            "recovery_count": self.recovery_count,
            "error_counts": {e.name: c for e, c in self.error_count.items() if c > 0},
            "gpu_healthy": self._gpu_healthy,
        }

class SafeHardwareContext:

    def __init__(self, handler: HardwareRecoveryHandler):
        self.handler = handler

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            error_type = self.handler.classify_error(exc_val)
            if error_type is not None:
                result = self.handler.try_recover(exc_val)
                if result.success:
                    return True
        return False

def hardware_safe(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                error_str = str(e).lower()
                if 'cuda' in error_str or 'device' in error_str:
                    if attempt < max_retries - 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        time.sleep(1)
                        continue
                raise

        raise RuntimeError(f"Hardware error in {func.__name__} after {max_retries} retries")

    return wrapper

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_info() -> Dict[str, Any]:
    info = {
        "best_device": str(get_best_device()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
            })

    if hasattr(torch.backends, 'mps'):
        info["mps_available"] = torch.backends.mps.is_available()

    return info