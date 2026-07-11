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

from __future__ import annotations

from datetime import timedelta
import inspect
import os
import sys

import pytest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_hardware_handler_does_not_use_torch_distributed_timedelta() -> None:
    import arc.intervention.hardware_handler as hh

    src = inspect.getsource(hh)
    assert "torch.distributed.timedelta" not in src


def test_recover_network_failure_barrier_timeout_is_datetime_timedelta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arc.intervention.hardware_handler import HardwareRecoveryHandler, HardwareConfig

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    handler = HardwareRecoveryHandler(
        model,
        optimizer,
        config=HardwareConfig(max_retries=1, retry_delay_seconds=0.0, verbose=False),
    )

    # Simulate dist being initialized, and capture the barrier timeout.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    # Ensure access to a nonexistent API would crash (as in real PyTorch).
    if hasattr(torch.distributed, "timedelta"):
        monkeypatch.delattr(torch.distributed, "timedelta", raising=False)

    captured = {}

    def fake_barrier(*, timeout=None, **kwargs):
        captured["timeout"] = timeout

    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)

    # Avoid real sleeping.
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    result = handler._recover_network_failure()

    assert result.success is True
    assert isinstance(captured.get("timeout"), timedelta)
    assert captured["timeout"].total_seconds() == 10


def test_recover_ddp_failure_barrier_timeout_is_datetime_timedelta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arc.intervention.hardware_handler import HardwareRecoveryHandler, HardwareConfig

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    handler = HardwareRecoveryHandler(
        model,
        optimizer,
        config=HardwareConfig(ddp_timeout_seconds=123.0, verbose=False),
    )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    if hasattr(torch.distributed, "timedelta"):
        monkeypatch.delattr(torch.distributed, "timedelta", raising=False)

    captured = {}

    def fake_barrier(*, timeout=None, **kwargs):
        captured["timeout"] = timeout

    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)

    result = handler._recover_ddp_failure()

    assert result.success is True
    assert isinstance(captured.get("timeout"), timedelta)
    assert captured["timeout"].total_seconds() == 123
