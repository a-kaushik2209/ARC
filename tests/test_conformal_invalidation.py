import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from arc.uncertainty.conformal import ConformalPredictor, ConformalRegression


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# ── ConformalPredictor tests ──────────────────────────────────────────────────

def test_conformal_predictor_pending_false_on_init():
    """recalibration_pending must default to False on init."""
    model = SimpleModel()
    cp = ConformalPredictor(model)
    assert cp.recalibration_pending is False


def test_conformal_predictor_invalidate_clears_scores():
    """Calibration scores must be cleared after invalidation."""
    model = SimpleModel()
    cp = ConformalPredictor(model)
    cp._calibration_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    cp.invalidate_calibration()
    assert cp._calibration_scores == []


def test_conformal_predictor_invalidate_clears_quantile():
    """Cached quantile must be None after invalidation."""
    model = SimpleModel()
    cp = ConformalPredictor(model)
    cp._quantile = 0.95
    cp.invalidate_calibration()
    assert cp._quantile is None


def test_conformal_predictor_invalidate_sets_uncalibrated():
    """_is_calibrated must be False after invalidation."""
    model = SimpleModel()
    cp = ConformalPredictor(model)
    cp._is_calibrated = True
    cp.invalidate_calibration()
    assert cp._is_calibrated is False


def test_conformal_predictor_recalibration_pending_set():
    """recalibration_pending flag must be True after invalidation."""
    model = SimpleModel()
    cp = ConformalPredictor(model)
    cp.invalidate_calibration()
    assert cp.recalibration_pending is True


def test_conformal_predictor_double_invalidation_safe():
    """Calling invalidate_calibration twice should not raise."""
    model = SimpleModel()
    cp = ConformalPredictor(model)
    cp._calibration_scores = [0.1, 0.2]
    cp.invalidate_calibration()
    cp.invalidate_calibration()
    assert cp._calibration_scores == []
    assert cp.recalibration_pending is True


# ── ConformalRegression tests ─────────────────────────────────────────────────

def test_conformal_regression_pending_false_on_init():
    """recalibration_pending must default to False on init."""
    model = SimpleModel()
    cr = ConformalRegression(model)
    assert cr.recalibration_pending is False


def test_conformal_regression_invalidate_clears_quantile():
    """Cached quantile must be None after invalidation."""
    model = SimpleModel()
    cr = ConformalRegression(model)
    cr._quantile = 0.8
    cr.invalidate_calibration()
    assert cr._quantile is None


def test_conformal_regression_invalidate_sets_uncalibrated():
    """_is_calibrated must be False after invalidation."""
    model = SimpleModel()
    cr = ConformalRegression(model)
    cr._is_calibrated = True
    cr.invalidate_calibration()
    assert cr._is_calibrated is False


def test_conformal_regression_recalibration_pending_set():
    """recalibration_pending flag must be True after invalidation."""
    model = SimpleModel()
    cr = ConformalRegression(model)
    cr.invalidate_calibration()
    assert cr.recalibration_pending is True


def test_conformal_regression_double_invalidation_safe():
    """Calling invalidate_calibration twice should not raise."""
    model = SimpleModel()
    cr = ConformalRegression(model)
    cr._quantile = 0.5
    cr.invalidate_calibration()
    cr.invalidate_calibration()
    assert cr._quantile is None
    assert cr.recalibration_pending is True


# ── SelfHealingArc integration tests ─────────────────────────────────────────

def test_selfhealing_rollback_invalidates_conformal_predictor():
    """Rollback must trigger invalidate_calibration on attached conformal_predictor."""
    from arc.core.self_healing import SelfHealingArc

    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    arc = SelfHealingArc(model, optimizer)

    cp = ConformalPredictor(model)
    cp._calibration_scores = [0.1, 0.2, 0.3]
    cp._quantile = 0.9
    cp._is_calibrated = True
    arc.conformal_predictor = cp

    arc._restore_checkpoint()

    assert arc.conformal_predictor._calibration_scores == []
    assert arc.conformal_predictor._quantile is None
    assert arc.conformal_predictor._is_calibrated is False
    assert arc.conformal_predictor.recalibration_pending is True


def test_selfhealing_rollback_invalidates_conformal_regressor():
    """Rollback must trigger invalidate_calibration on attached conformal_regressor."""
    from arc.core.self_healing import SelfHealingArc

    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    arc = SelfHealingArc(model, optimizer)

    cr = ConformalRegression(model)
    cr._quantile = 0.85
    cr._is_calibrated = True
    arc.conformal_regressor = cr

    arc._restore_checkpoint()

    assert arc.conformal_regressor._quantile is None
    assert arc.conformal_regressor._is_calibrated is False
    assert arc.conformal_regressor.recalibration_pending is True


def test_selfhealing_rollback_safe_without_conformal():
    """Rollback must not raise if conformal modules are not attached."""
    from arc.core.self_healing import SelfHealingArc

    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    arc = SelfHealingArc(model, optimizer)

    # Should not raise
    arc._restore_checkpoint()