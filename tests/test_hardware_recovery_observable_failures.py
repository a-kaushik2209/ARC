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

import inspect
import os
import sys

import pytest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_no_silent_exception_suppression_in_key_recovery_paths() -> None:
    from arc.intervention.hardware_handler import HardwareRecoveryHandler

    disk_src = inspect.getsource(HardwareRecoveryHandler._recover_disk_full)
    ddp_src = inspect.getsource(HardwareRecoveryHandler._recover_ddp_failure)

    # Guard against the specific anti-pattern that caused Issue #36.
    assert "except Exception" not in disk_src
    assert "pass" not in disk_src

    assert "except Exception" not in ddp_src
    assert "pass" not in ddp_src


def test_disk_recovery_failure_emits_warning_and_includes_details(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from arc.intervention.hardware_handler import HardwareRecoveryHandler, HardwareConfig

    # Create >2 checkpoint files so cleanup loop attempts at least one remove.
    for name in ["a.pt", "b.pt", "c.pt"]:
        (tmp_path / name).write_bytes(b"x")

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    config = HardwareConfig(
        checkpoint_dir=str(tmp_path),
        remote_checkpoint_url="https://example.invalid/checkpoints",
        verbose=False,
    )
    handler = HardwareRecoveryHandler(model, optimizer, config=config)

    def failing_remove(_path: str) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(os, "remove", failing_remove)

    with pytest.warns(UserWarning, match="Disk cleanup failed during recovery"):
        result = handler._recover_disk_full()

    assert result.details.get("cleanup_error")
    assert "permission denied" in result.details["cleanup_error"]


def test_ddp_recovery_failure_emits_warning_and_includes_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arc.intervention.hardware_handler import HardwareRecoveryHandler, HardwareConfig

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    handler = HardwareRecoveryHandler(model, optimizer, config=HardwareConfig(verbose=False))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def failing_barrier(*, timeout=None, **kwargs):
        raise RuntimeError("barrier timed out")

    monkeypatch.setattr(torch.distributed, "barrier", failing_barrier)

    with pytest.warns(UserWarning, match="DDP barrier failed during recovery"):
        result = handler._recover_ddp_failure()

    assert result.success is True
    assert result.details.get("barrier_error")
    assert "barrier timed out" in result.details["barrier_error"]
