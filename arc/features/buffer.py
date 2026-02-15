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

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Iterator
from collections import deque
import numpy as np
import json

@dataclass
class EpochSnapshot:
    epoch: int
    step: int
    signals: Dict[str, Any]
    timestamp: float

    def get_flat_signals(self, prefix: str = "") -> Dict[str, float]:
        flat = {}
        self._flatten_recursive(self.signals, prefix, flat)
        return flat

    def _flatten_recursive(
        self,
        obj: Any,
        prefix: str,
        result: Dict[str, float]
    ) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._flatten_recursive(value, new_prefix, result)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if np.isfinite(obj):
                result[prefix] = float(obj)
        elif isinstance(obj, np.ndarray):
            if obj.size == 1:
                val = obj.item()
                if np.isfinite(val):
                    result[prefix] = float(val)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "signals": self.signals,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EpochSnapshot':
        return cls(
            epoch=d["epoch"],
            step=d["step"],
            signals=d["signals"],
            timestamp=d["timestamp"],
        )

class SignalBuffer:

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._signal_keys: Optional[set] = None

    def append(self, snapshot: EpochSnapshot) -> None:
        self._buffer.append(snapshot)

        if self._signal_keys is None:
            self._signal_keys = set(snapshot.get_flat_signals().keys())
        else:
            self._signal_keys.update(snapshot.get_flat_signals().keys())

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, index: int) -> EpochSnapshot:
        return self._buffer[index]

    def __iter__(self) -> Iterator[EpochSnapshot]:
        return iter(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.max_size

    @property
    def epochs(self) -> List[int]:
        return [s.epoch for s in self._buffer]

    @property
    def signal_keys(self) -> set:
        return self._signal_keys or set()

    def get_recent(self, n: int = 1) -> List[EpochSnapshot]:
        n = min(n, len(self._buffer))
        return list(self._buffer)[-n:]

    def get_signal_history(
        self,
        key: str,
        n_epochs: Optional[int] = None,
        default: float = 0.0
    ) -> np.ndarray:
        snapshots = self.get_recent(n_epochs) if n_epochs else list(self._buffer)

        values = []
        for snapshot in snapshots:
            flat = snapshot.get_flat_signals()
            values.append(flat.get(key, default))

        return np.array(values)

    def get_all_signals_matrix(
        self,
        keys: Optional[List[str]] = None,
        n_epochs: Optional[int] = None
    ) -> tuple:
        snapshots = self.get_recent(n_epochs) if n_epochs else list(self._buffer)

        if not snapshots:
            return np.array([]), [], []

        if keys is None:
            keys = sorted(self._signal_keys) if self._signal_keys else []

        if not keys:
            return np.array([]), [], []

        matrix = np.zeros((len(snapshots), len(keys)))
        epochs = []

        for i, snapshot in enumerate(snapshots):
            flat = snapshot.get_flat_signals()
            epochs.append(snapshot.epoch)
            for j, key in enumerate(keys):
                matrix[i, j] = flat.get(key, np.nan)

        return matrix, keys, epochs

    def get_windowed_signals(
        self,
        key: str,
        window_size: int,
        stride: int = 1
    ) -> List[np.ndarray]:
        full_history = self.get_signal_history(key)

        if len(full_history) < window_size:
            return []

        windows = []
        for i in range(0, len(full_history) - window_size + 1, stride):
            windows.append(full_history[i:i + window_size])

        return windows

    def clear(self) -> None:
        self._buffer.clear()
        self._signal_keys = None

    def to_json(self) -> str:
        data = {
            "max_size": self.max_size,
            "snapshots": [s.to_dict() for s in self._buffer],
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'SignalBuffer':
        data = json.loads(json_str)
        buffer = cls(max_size=data["max_size"])
        for s_dict in data["snapshots"]:
            buffer.append(EpochSnapshot.from_dict(s_dict))
        return buffer

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'SignalBuffer':
        with open(path, 'r') as f:
            return cls.from_json(f.read())
