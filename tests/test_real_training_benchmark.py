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
Tests for the Real Training Failure Benchmark Suite (issue #61).
"""

import json
import os
import tempfile

import pytest
import torch

from arc.evaluation.real_training_benchmark import (
    RealTrainingFailureBenchmark,
    RealTrainingBenchmarkResult,
    _make_model,
    _make_data,
)


@pytest.fixture
def suite():
    """Lightweight suite: fewer epochs so tests run fast."""
    return RealTrainingFailureBenchmark(n_epochs=12, inject_at_epoch=5, verbose=False)


# ---------------------------------------------------------------------------
# Smoke tests — each benchmark returns the right type without crashing
# ---------------------------------------------------------------------------

def test_exploding_gradients_returns_result(suite):
    result = suite.benchmark_exploding_gradients()
    assert isinstance(result, RealTrainingBenchmarkResult)
    assert result.name == "Exploding Gradients"


def test_nan_generation_returns_result(suite):
    result = suite.benchmark_nan_generation()
    assert isinstance(result, RealTrainingBenchmarkResult)
    assert result.name == "NaN Generation"


def test_fp16_overflow_returns_result(suite):
    result = suite.benchmark_fp16_overflow()
    assert isinstance(result, RealTrainingBenchmarkResult)
    assert result.name == "fp16 Overflow"


def test_optimizer_corruption_returns_result(suite):
    result = suite.benchmark_optimizer_corruption()
    assert isinstance(result, RealTrainingBenchmarkResult)
    assert result.name == "Optimizer State Corruption"


def test_checkpoint_corruption_returns_result(suite):
    result = suite.benchmark_checkpoint_corruption()
    assert isinstance(result, RealTrainingBenchmarkResult)
    assert result.name == "Checkpoint Corruption"


def test_dataloader_failure_returns_result(suite):
    result = suite.benchmark_dataloader_failure()
    assert isinstance(result, RealTrainingBenchmarkResult)
    assert result.name == "Dataloader Failure"


def test_healthy_baseline_returns_result(suite):
    result = suite.benchmark_healthy_baseline()
    assert isinstance(result, RealTrainingBenchmarkResult)
    assert result.name == "Healthy Baseline (FPR)"


# ---------------------------------------------------------------------------
# Detection assertions
# ---------------------------------------------------------------------------

def test_nan_generation_is_detected(suite):
    result = suite.benchmark_nan_generation()
    assert result.failure_detected, "NaN injection must be detected"


def test_checkpoint_corruption_is_detected(suite):
    result = suite.benchmark_checkpoint_corruption()
    assert result.failure_detected, "Corrupt checkpoint must raise an error"


def test_dataloader_failure_is_detected(suite):
    result = suite.benchmark_dataloader_failure()
    assert result.failure_detected, "Dataloader crash must be caught"


def test_dataloader_failure_recovery(suite):
    result = suite.benchmark_dataloader_failure()
    assert result.recovery_success, "Training should continue after dataloader failure"


def test_checkpoint_recovery(suite):
    result = suite.benchmark_checkpoint_corruption()
    assert result.recovery_success, "Should fall back to a fresh model"


# ---------------------------------------------------------------------------
# Healthy baseline: no false positives on clean run
# ---------------------------------------------------------------------------

def test_healthy_baseline_no_false_positives(suite):
    result = suite.benchmark_healthy_baseline()
    assert result.false_positive_rate == 0.0, (
        f"Healthy run should have 0 false positives, got {result.false_positive_rate}"
    )


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

def test_result_to_dict(suite):
    result = suite.benchmark_nan_generation()
    d = result.to_dict()
    assert "name" in d
    assert "failure_detected" in d
    assert "detection_latency_epochs" in d
    assert "false_positive_rate" in d
    assert "overhead_percent" in d


def test_save_results(suite):
    results = [
        suite.benchmark_nan_generation(),
        suite.benchmark_healthy_baseline(),
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        suite.save_results(results, path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["summary"]["total"] == 2
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# run_all smoke test
# ---------------------------------------------------------------------------

def test_run_all_returns_all_benchmarks(suite):
    results = suite.run_all()
    assert len(results) == 7
    names = {r.name for r in results}
    assert "Exploding Gradients" in names
    assert "NaN Generation" in names
    assert "fp16 Overflow" in names
    assert "Optimizer State Corruption" in names
    assert "Checkpoint Corruption" in names
    assert "Dataloader Failure" in names
    assert "Healthy Baseline (FPR)" in names


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def test_make_model_is_deterministic():
    m1 = _make_model(seed=0)
    m2 = _make_model(seed=0)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_make_data_shape():
    x, y = _make_data(n=128)
    assert x.shape == (128, 64)
    assert y.shape == (128,)
