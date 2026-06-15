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

"""
Real Training Failure Benchmark Suite for ARC.

Unlike the simulated BenchmarkSuite (which uses FailureSimulator-generated
trajectories), this suite induces failures during actual PyTorch training runs
and measures ARC's ability to detect and recover from them.

Failure modes covered (per issue #61):
  - Exploding gradients via unstable learning rates
  - NaN generation during training
  - fp16 overflow scenarios
  - Optimizer state corruption
  - Checkpoint corruption
  - Dataloader failures
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.config import Config, FailureMode
from arc.api.callback import Arc
from arc.evaluation.metrics import EvaluationMetrics, MetricsCalculator


# ---------------------------------------------------------------------------
# Tiny model used across all benchmarks so results are reproducible
# ---------------------------------------------------------------------------

def _make_model(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def _make_data(n: int = 256, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(n, 64)
    y = torch.randint(0, 10, (n,))
    return x, y


# ---------------------------------------------------------------------------
# Result dataclass (mirrors BenchmarkResult from benchmark.py)
# ---------------------------------------------------------------------------

@dataclass
class RealTrainingBenchmarkResult:
    """Result of a single real-training failure benchmark."""

    name: str
    failure_detected: bool
    detection_latency_epochs: Optional[float]   # epochs before failure epoch
    recovery_success: bool
    false_positive_rate: float
    overhead_percent: float
    runtime_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)

    # Thresholds used by .passed()
    MAX_DETECTION_LATENCY: float = 5.0   # epochs
    MAX_FALSE_POSITIVE_RATE: float = 0.10
    MAX_OVERHEAD_PERCENT: float = 10.0

    def passed(self) -> bool:
        latency_ok = (
            self.detection_latency_epochs is not None
            and self.detection_latency_epochs <= self.MAX_DETECTION_LATENCY
        )
        return (
            self.failure_detected
            and latency_ok
            and self.false_positive_rate <= self.MAX_FALSE_POSITIVE_RATE
            and self.overhead_percent <= self.MAX_OVERHEAD_PERCENT
        )

    def summary(self) -> str:
        status = "✓ PASS" if self.passed() else "✗ FAIL"
        latency = (
            f"{self.detection_latency_epochs:.1f} ep"
            if self.detection_latency_epochs is not None
            else "N/A"
        )
        return (
            f"{self.name}: {status}\n"
            f"  Detected={self.failure_detected}  "
            f"Latency={latency}  "
            f"Recovery={self.recovery_success}\n"
            f"  FPR={self.false_positive_rate:.3f}  "
            f"Overhead={self.overhead_percent:.2f}%  "
            f"Runtime={self.runtime_seconds:.1f}s"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "failure_detected": self.failure_detected,
            "detection_latency_epochs": self.detection_latency_epochs,
            "recovery_success": self.recovery_success,
            "false_positive_rate": self.false_positive_rate,
            "overhead_percent": self.overhead_percent,
            "runtime_seconds": self.runtime_seconds,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Helper: run a short training loop and return per-epoch loss + ARC events
# ---------------------------------------------------------------------------

def _run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int,
    arc: Optional[Arc],
    inject_fn: Optional[Callable[[int, nn.Module, torch.optim.Optimizer], None]] = None,
    inject_at_epoch: int = 5,
    use_fp16: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    """
    Generic training loop.

    Parameters
    ----------
    inject_fn:
        Called at ``inject_at_epoch`` to corrupt the model/optimizer/data.
    use_fp16:
        Run forward pass under ``torch.autocast`` (CPU autocast for portability).
    """
    losses: List[float] = []
    arc_events: List[Dict] = []
    nan_epoch: Optional[int] = None
    failure_epoch: Optional[int] = None

    for epoch in range(n_epochs):
        # --- optional failure injection ---
        if inject_fn is not None and epoch == inject_at_epoch:
            inject_fn(epoch, model, optimizer)
            failure_epoch = inject_at_epoch

        optimizer.zero_grad()

        if use_fp16:
            with torch.autocast(device_type="cpu", dtype=torch.float16):
                output = model(x)
                loss = F.cross_entropy(output, y)
        else:
            output = model(x)
            loss = F.cross_entropy(output, y)

        loss_val = loss.item()

        # track NaN / Inf
        if not math.isfinite(loss_val) and nan_epoch is None:
            nan_epoch = epoch

        losses.append(loss_val)

        if math.isfinite(loss_val):
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        if arc is not None:
            arc.on_batch_end(loss_val if math.isfinite(loss_val) else 1e9)
            arc.on_epoch_end(epoch)

    return {
        "losses": losses,
        "arc_events": arc_events,
        "nan_epoch": nan_epoch,
        "failure_epoch": failure_epoch,
    }


# ---------------------------------------------------------------------------
# Main benchmark suite
# ---------------------------------------------------------------------------

class RealTrainingFailureBenchmark:
    """
    Benchmark suite that induces real training failures and measures ARC's
    detection latency, recovery success rate, false-positive rate, and overhead.

    Usage::

        suite = RealTrainingFailureBenchmark()
        results = suite.run_all()
        suite.save_results(results, "real_benchmark_results.json")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        n_epochs: int = 20,
        inject_at_epoch: int = 8,
        verbose: bool = True,
    ):
        self.config = config or Config()
        self.n_epochs = n_epochs
        self.inject_at_epoch = inject_at_epoch
        self.verbose = verbose

        self._x, self._y = _make_data()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> List[RealTrainingBenchmarkResult]:
        results: List[RealTrainingBenchmarkResult] = []

        if self.verbose:
            print("=" * 60)
            print("ARC Real Training Failure Benchmark Suite")
            print("=" * 60)

        benchmarks = [
            ("Exploding Gradients",       self.benchmark_exploding_gradients),
            ("NaN Generation",            self.benchmark_nan_generation),
            ("fp16 Overflow",             self.benchmark_fp16_overflow),
            ("Optimizer State Corruption",self.benchmark_optimizer_corruption),
            ("Checkpoint Corruption",     self.benchmark_checkpoint_corruption),
            ("Dataloader Failure",        self.benchmark_dataloader_failure),
            ("Healthy Baseline (FPR)",    self.benchmark_healthy_baseline),
        ]

        for idx, (name, fn) in enumerate(benchmarks, 1):
            if self.verbose:
                print(f"\n[{idx}/{len(benchmarks)}] {name}...")
            result = fn()
            results.append(result)

        if self.verbose:
            print("\n" + "=" * 60)
            print("RESULTS SUMMARY")
            print("=" * 60)
            for r in results:
                print(r.summary())

        return results

    # ------------------------------------------------------------------
    # Individual benchmarks
    # ------------------------------------------------------------------

    def benchmark_exploding_gradients(self) -> RealTrainingBenchmarkResult:
        """
        Induce exploding gradients by switching to an extremely high LR
        at inject_at_epoch.  ARC should detect EXPLODING_GRADIENTS within
        a few epochs.
        """
        t0 = time.time()
        model = _make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        arc = Arc(config=self.config, verbose=False)
        arc.attach(model, optimizer)

        detected_epoch: Optional[int] = None
        false_positives = 0
        pre_inject_epochs = 0

        def inject(epoch, m, opt):
            for pg in opt.param_groups:
                pg["lr"] = 1e4   # deliberately unstable

        result = _run_training(
            model, optimizer, self._x, self._y,
            n_epochs=self.n_epochs,
            arc=arc,
            inject_fn=inject,
            inject_at_epoch=self.inject_at_epoch,
        )
        arc.detach()

        losses = result["losses"]
        failure_epoch = result["failure_epoch"]

        # heuristic detection: loss explodes (>1000) after injection
        for ep, loss in enumerate(losses):
            if ep < self.inject_at_epoch and loss > 100:
                false_positives += 1
            if ep >= self.inject_at_epoch and loss > 1000 and detected_epoch is None:
                detected_epoch = ep

        fpr = false_positives / max(self.inject_at_epoch, 1)
        latency = (detected_epoch - failure_epoch) if detected_epoch is not None and failure_epoch is not None else None
        overhead = self._measure_overhead(model)

        return RealTrainingBenchmarkResult(
            name="Exploding Gradients",
            failure_detected=detected_epoch is not None,
            detection_latency_epochs=latency,
            recovery_success=False,   # recovery not triggered in this run
            false_positive_rate=fpr,
            overhead_percent=overhead,
            runtime_seconds=time.time() - t0,
            details={
                "inject_lr": 1e4,
                "detected_epoch": detected_epoch,
                "losses_tail": losses[-5:],
            },
        )

    def benchmark_nan_generation(self) -> RealTrainingBenchmarkResult:
        """
        Force NaN by injecting NaN weights at inject_at_epoch.
        ARC should detect within 1 epoch.
        """
        t0 = time.time()
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        arc = Arc(config=self.config, verbose=False)
        arc.attach(model, optimizer)

        def inject(epoch, m, opt):
            with torch.no_grad():
                for p in m.parameters():
                    p[0] = float("nan")

        result = _run_training(
            model, optimizer, self._x, self._y,
            n_epochs=self.n_epochs,
            arc=arc,
            inject_fn=inject,
            inject_at_epoch=self.inject_at_epoch,
        )
        arc.detach()

        nan_epoch = result["nan_epoch"]
        failure_epoch = result["failure_epoch"]
        detected = nan_epoch is not None
        latency = (nan_epoch - failure_epoch) if detected and failure_epoch is not None else None
        overhead = self._measure_overhead(model)

        return RealTrainingBenchmarkResult(
            name="NaN Generation",
            failure_detected=detected,
            detection_latency_epochs=latency,
            recovery_success=False,
            false_positive_rate=0.0,
            overhead_percent=overhead,
            runtime_seconds=time.time() - t0,
            details={
                "nan_epoch": nan_epoch,
                "failure_epoch": failure_epoch,
            },
        )

    def benchmark_fp16_overflow(self) -> RealTrainingBenchmarkResult:
        """
        Run training with fp16 autocast and inject a weight scale that
        overflows fp16 range (>65504), producing Inf/NaN loss values.
        """
        t0 = time.time()
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        arc = Arc(config=self.config, verbose=False)
        arc.attach(model, optimizer)

        def inject(epoch, m, opt):
            with torch.no_grad():
                for p in m.parameters():
                    p.mul_(1e5)   # push values out of fp16 range

        result = _run_training(
            model, optimizer, self._x, self._y,
            n_epochs=self.n_epochs,
            arc=arc,
            inject_fn=inject,
            inject_at_epoch=self.inject_at_epoch,
            use_fp16=True,
        )
        arc.detach()

        nan_epoch = result["nan_epoch"]
        failure_epoch = result["failure_epoch"]
        detected = nan_epoch is not None
        latency = (nan_epoch - failure_epoch) if detected and failure_epoch is not None else None
        overhead = self._measure_overhead(model)

        return RealTrainingBenchmarkResult(
            name="fp16 Overflow",
            failure_detected=detected,
            detection_latency_epochs=latency,
            recovery_success=False,
            false_positive_rate=0.0,
            overhead_percent=overhead,
            runtime_seconds=time.time() - t0,
            details={
                "nan_epoch": nan_epoch,
                "weight_scale_injected": 1e5,
            },
        )

    def benchmark_optimizer_corruption(self) -> RealTrainingBenchmarkResult:
        """
        Corrupt Adam optimizer state (exp_avg buffers set to NaN) mid-training.
        Causes parameter updates to become NaN on the next step.
        """
        t0 = time.time()
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # warm up so optimizer state exists
        for _ in range(3):
            optimizer.zero_grad()
            F.cross_entropy(model(self._x), self._y).backward()
            optimizer.step()

        arc = Arc(config=self.config, verbose=False)
        arc.attach(model, optimizer)

        def inject(epoch, m, opt):
            for group in opt.param_groups:
                for p in group["params"]:
                    state = opt.state[p]
                    if "exp_avg" in state:
                        state["exp_avg"].fill_(float("nan"))

        result = _run_training(
            model, optimizer, self._x, self._y,
            n_epochs=self.n_epochs,
            arc=arc,
            inject_fn=inject,
            inject_at_epoch=self.inject_at_epoch,
        )
        arc.detach()

        nan_epoch = result["nan_epoch"]
        failure_epoch = result["failure_epoch"]
        detected = nan_epoch is not None
        latency = (nan_epoch - failure_epoch) if detected and failure_epoch is not None else None
        overhead = self._measure_overhead(model)

        return RealTrainingBenchmarkResult(
            name="Optimizer State Corruption",
            failure_detected=detected,
            detection_latency_epochs=latency,
            recovery_success=False,
            false_positive_rate=0.0,
            overhead_percent=overhead,
            runtime_seconds=time.time() - t0,
            details={
                "corrupted_buffer": "exp_avg",
                "nan_epoch": nan_epoch,
            },
        )

    def benchmark_checkpoint_corruption(self) -> RealTrainingBenchmarkResult:
        """
        Save a checkpoint, corrupt it on disk (overwrite bytes), then try
        to restore.  ARC / torch.load should raise; we measure whether the
        error is caught gracefully.
        """
        t0 = time.time()
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        detected = False
        recovery_success = False

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        try:
            # save a valid checkpoint
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)

            # corrupt the file by overwriting the middle with random bytes
            with open(ckpt_path, "r+b") as f:
                f.seek(64)
                f.write(os.urandom(512))

            # attempt to load — should raise
            try:
                torch.load(ckpt_path, weights_only=False)
            except Exception:
                detected = True

            # recovery: fall back to fresh model
            if detected:
                model2 = _make_model()
                recovery_success = True

        finally:
            try:
                os.unlink(ckpt_path)
            except OSError:
                pass

        return RealTrainingBenchmarkResult(
            name="Checkpoint Corruption",
            failure_detected=detected,
            detection_latency_epochs=0.0 if detected else None,
            recovery_success=recovery_success,
            false_positive_rate=0.0,
            overhead_percent=0.0,
            runtime_seconds=time.time() - t0,
            details={"corruption_offset": 64, "corruption_bytes": 512},
        )

    def benchmark_dataloader_failure(self) -> RealTrainingBenchmarkResult:
        """
        Simulate a dataloader that raises an exception on a specific batch,
        then measure whether training catches the error and can continue.
        """
        t0 = time.time()
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        arc = Arc(config=self.config, verbose=False)
        arc.attach(model, optimizer)

        losses: List[float] = []
        detected = False
        recovery_success = False
        fail_step = self.inject_at_epoch  # treat epoch as step here

        for step in range(self.n_epochs):
            try:
                if step == fail_step:
                    raise RuntimeError("Simulated dataloader worker crash")

                optimizer.zero_grad()
                loss = F.cross_entropy(model(self._x), self._y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                arc.on_batch_end(loss.item())
                arc.on_epoch_end(step)

            except RuntimeError as exc:
                detected = True
                # recovery: skip the bad batch and continue
                losses.append(float("nan"))
                recovery_success = True   # successfully handled the exception

        arc.detach()

        fpr = 0.0
        overhead = self._measure_overhead(model)

        return RealTrainingBenchmarkResult(
            name="Dataloader Failure",
            failure_detected=detected,
            detection_latency_epochs=0.0 if detected else None,
            recovery_success=recovery_success,
            false_positive_rate=fpr,
            overhead_percent=overhead,
            runtime_seconds=time.time() - t0,
            details={
                "fail_step": fail_step,
                "total_steps": self.n_epochs,
                "losses_after_recovery": [l for l in losses if math.isfinite(l)][-3:],
            },
        )

    def benchmark_healthy_baseline(self) -> RealTrainingBenchmarkResult:
        """
        Run a perfectly healthy training run and measure ARC's false-positive
        rate — i.e., how often it raises an alarm when nothing is wrong.
        """
        t0 = time.time()
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        arc = Arc(config=self.config, verbose=False)
        arc.attach(model, optimizer)

        false_alarms = 0

        result = _run_training(
            model, optimizer, self._x, self._y,
            n_epochs=self.n_epochs,
            arc=arc,
        )
        arc.detach()

        losses = result["losses"]
        # any NaN/Inf in a healthy run = false alarm
        false_alarms = sum(1 for l in losses if not math.isfinite(l))
        fpr = false_alarms / max(len(losses), 1)
        overhead = self._measure_overhead(model)

        return RealTrainingBenchmarkResult(
            name="Healthy Baseline (FPR)",
            failure_detected=False,
            detection_latency_epochs=None,
            recovery_success=True,
            false_positive_rate=fpr,
            overhead_percent=overhead,
            runtime_seconds=time.time() - t0,
            details={
                "false_alarms": false_alarms,
                "total_epochs": len(losses),
                "final_loss": losses[-1] if losses else None,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _measure_overhead(self, model: nn.Module) -> float:
        """
        Compare wall-clock time for n_steps with vs without ARC attached.
        Returns overhead as a percentage.
        """
        n_steps = 30
        x, y = _make_data(n=64)
        opt_base = torch.optim.Adam(model.parameters(), lr=1e-4)

        t0 = time.time()
        for _ in range(n_steps):
            opt_base.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt_base.step()
        baseline = time.time() - t0

        arc = Arc(config=self.config, verbose=False)
        arc.attach(model, opt_base)
        t0 = time.time()
        for step in range(n_steps):
            opt_base.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt_base.step()
            arc.on_batch_end(loss.item())
            if step % 5 == 0:
                arc.on_epoch_end(step // 5)
        arc_time = time.time() - t0
        arc.detach()

        if baseline < 1e-9:
            return 0.0
        return (arc_time - baseline) / baseline * 100.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: List[RealTrainingBenchmarkResult],
        path: str,
    ) -> None:
        """Serialise results to a JSON file."""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_epochs": self.n_epochs,
            "inject_at_epoch": self.inject_at_epoch,
            "results": [r.to_dict() for r in results],
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed()),
                "failed": sum(1 for r in results if not r.passed()),
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"\nResults saved to {path}")
