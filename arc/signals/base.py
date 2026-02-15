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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import time
import functools
import torch
import torch.nn as nn

@dataclass
class SignalSnapshot:

    step: int
    epoch: Optional[int]
    timestamp: float
    signals: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:

        return self.signals.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.signals[key]

    def keys(self):
        return self.signals.keys()

    def values(self):
        return self.signals.values()

    def items(self):
        return self.signals.items()

def measure_overhead(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        if hasattr(self, 'overhead_history'):
            self.overhead_history.append(elapsed)
        return result
    return wrapper

class SignalCollector(ABC):

    def __init__(self, config: Optional[Any] = None):

        self.config = config
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._hooks: List[Any] = []
        self._is_attached: bool = False
        self._step: int = 0
        self._epoch: int = 0
        self.overhead_history: List[float] = []

    def attach(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> None:

        if self._is_attached:
            self.detach()

        self._model = model
        self._optimizer = optimizer
        self._register_hooks()
        self._is_attached = True

    def detach(self) -> None:

        self._cleanup_hooks()
        self._hooks.clear()
        self._model = None
        self._optimizer = None
        self._is_attached = False

    def reset(self) -> None:

        self._step = 0
        self.overhead_history.clear()

    def set_epoch(self, epoch: int) -> None:

        self._epoch = epoch

    def step(self) -> None:

        self._step += 1

    @measure_overhead
    def collect(self) -> SignalSnapshot:

        if not self._is_attached:
            raise RuntimeError("Collector not attached. Call attach() first.")

        signals = self._collect_signals()

        return SignalSnapshot(
            step=self._step,
            epoch=self._epoch,
            timestamp=time.time(),
            signals=signals,
            metadata=self._get_metadata()
        )

    def get_average_overhead(self) -> float:

        if not self.overhead_history:
            return 0.0
        return sum(self.overhead_history) / len(self.overhead_history)

    @abstractmethod
    def _register_hooks(self) -> None:

        pass

    @abstractmethod
    def _collect_signals(self) -> Dict[str, Any]:

        pass

    def _cleanup_hooks(self) -> None:

        for hook in self._hooks:
            hook.remove()

    def _get_metadata(self) -> Dict[str, Any]:

        return {
            "collector_type": self.__class__.__name__,
            "is_attached": self._is_attached,
        }

    @property
    def is_attached(self) -> bool:

        return self._is_attached

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.detach()
        return False

class CompositeCollector(SignalCollector):

    def __init__(self, collectors: List[SignalCollector]):

        super().__init__()
        self.collectors = collectors

    def attach(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> None:

        for collector in self.collectors:
            collector.attach(model, optimizer)
        self._is_attached = True
        self._model = model
        self._optimizer = optimizer

    def detach(self) -> None:

        for collector in self.collectors:
            collector.detach()
        self._is_attached = False

    def reset(self) -> None:

        for collector in self.collectors:
            collector.reset()

    def set_epoch(self, epoch: int) -> None:

        for collector in self.collectors:
            collector.set_epoch(epoch)

    def step(self) -> None:

        for collector in self.collectors:
            collector.step()

    def _register_hooks(self) -> None:

        pass

    def _collect_signals(self) -> Dict[str, Any]:

        merged = {}
        for collector in self.collectors:
            snapshot = collector.collect()
            prefix = collector.__class__.__name__.replace("Collector", "").lower()
            for key, value in snapshot.signals.items():
                merged[f"{prefix}.{key}"] = value
        return merged

    def _cleanup_hooks(self) -> None:

        pass

    def get_average_overhead(self) -> float:

        return sum(c.get_average_overhead() for c in self.collectors)
