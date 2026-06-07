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
ARC Test Suite

Comprehensive tests for all ARC components.
Run with: pytest tests/test_arc.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )


@pytest.fixture
def simple_cnn():
    """Simple CNN for testing."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.fc = nn.Linear(16 * 16 * 16, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    return SimpleCNN()


# =============================================================================
# WeightRollback Tests
# =============================================================================

class TestWeightRollback:
    """Tests for WeightRollback intervention system."""
    
    def test_import(self):
        """Test that WeightRollback can be imported."""
        from arc.intervention import WeightRollback
        assert WeightRollback is not None
    
    def test_initialization(self, simple_model):
        """Test WeightRollback initialization."""
        from arc.intervention import WeightRollback
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer)
        
        assert rollback is not None
        assert len(rollback.state.checkpoints) == 1  # Initial checkpoint
    
    def test_checkpoint_saving(self, simple_model):
        """Test that checkpoints are saved periodically."""
        from arc.intervention import WeightRollback, RollbackConfig
        
        config = RollbackConfig(checkpoint_frequency=5)
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer, config, verbose=False)
        
        # Run 10 steps
        for _ in range(10):
            x = torch.randn(4, 100)
            loss = simple_model(x).mean()
            loss.backward()
            rollback.step(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        # Should have initial + 2 periodic checkpoints
        assert len(rollback.state.checkpoints) >= 2
    
    def test_failure_detection_nan(self, simple_model):
        """Test NaN loss detection."""
        from arc.intervention import WeightRollback
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer, verbose=False)
        
        # Normal step
        action1 = rollback.step(torch.tensor(1.0))
        assert not action1.rolled_back
        
        # NaN step
        action2 = rollback.step(torch.tensor(float('nan')))
        assert action2.failure_detected
        assert action2.failure_type == "nan_loss"
    
    def test_rollback_restores_weights(self, simple_model):
        """Test that rollback actually restores weights."""
        from arc.intervention import WeightRollback, RollbackConfig

        config = RollbackConfig(checkpoint_frequency=1, loss_explosion_threshold=10.0)
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer, config, verbose=False)

        # Save initial weights before any steps
        initial_weights = simple_model[0].weight.data.clone()

        # Step 1: save a checkpoint with CLEAN weights
        rollback.step(torch.tensor(1.0))

        # Step 2: corrupt weights AFTER the checkpoint is saved
        with torch.no_grad():
            simple_model[0].weight.data.fill_(999.0)

        # Step 3: trigger rollback with an explosive loss
        action = rollback.step(torch.tensor(100.0))

        # Step 4: rollback MUST have fired — assert unconditionally
        assert action.rolled_back, "Expected rollback to trigger but it did not"

        # Step 5: weights must no longer be the corrupted value
        assert not torch.allclose(
            simple_model[0].weight.data,
            torch.full_like(simple_model[0].weight.data, 999.0)
        ), "Weights were not restored — still showing corrupted 999.0 values"

        # Step 6: weights must be back to the clean checkpoint values
        assert torch.allclose(
            simple_model[0].weight.data, initial_weights, atol=1e-5
        ), "Weights were not restored to the pre-corruption checkpoint values"

    def test_max_checkpoints_config_respected(self, simple_model):
        """Regression for issue #14: RollbackConfig.max_checkpoints must size the deque."""
        from arc.intervention import WeightRollback, RollbackConfig

        config = RollbackConfig(max_checkpoints=5)
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer, config, verbose=False)

        assert rollback.state.checkpoints.maxlen == 5

    def test_max_checkpoints_retains_more_than_three(self, simple_model):
        """With max_checkpoints=5, more than the old hardcoded 3 checkpoints are kept."""
        from arc.intervention import WeightRollback, RollbackConfig

        config = RollbackConfig(max_checkpoints=5, checkpoint_frequency=1)
        optimizer = torch.optim.Adam(simple_model.parameters())
        rollback = WeightRollback(simple_model, optimizer, config, verbose=False)

        for _ in range(10):
            rollback.step(torch.tensor(1.0))

        assert len(rollback.state.checkpoints) == 5


# =============================================================================
# GradientForecaster Tests
# =============================================================================

class TestGradientForecaster:
    """Tests for GradientForecaster prediction system."""
    
    def test_import(self):
        """Test that GradientForecaster can be imported."""
        from arc.prediction import GradientForecaster
        assert GradientForecaster is not None
    
    def test_initialization(self, simple_model):
        """Test GradientForecaster initialization."""
        from arc.prediction import GradientForecaster
        
        forecaster = GradientForecaster(simple_model)
        assert forecaster is not None
        assert forecaster.step_count == 0
    
    def test_update(self, simple_model):
        """Test forecaster update."""
        from arc.prediction import GradientForecaster
        
        forecaster = GradientForecaster(simple_model)
        
        x = torch.randn(4, 100)
        loss = simple_model(x).mean()
        loss.backward()
        
        update_info = forecaster.update()
        assert "grad_norm" in update_info
        assert update_info["step"] == 1
    
    def test_prediction(self, simple_model):
        """Test gradient prediction."""
        from arc.prediction import GradientForecaster
        
        forecaster = GradientForecaster(simple_model)
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        
        # Run a few steps to build history
        for _ in range(10):
            x = torch.randn(4, 100)
            loss = simple_model(x).mean()
            optimizer.zero_grad()
            loss.backward()
            forecaster.update()
            optimizer.step()
        
        forecast = forecaster.predict()
        assert hasattr(forecast, 'will_explode')
        assert hasattr(forecast, 'steps_until')
        assert hasattr(forecast, 'confidence')


# =============================================================================
# LiteArc Tests
# =============================================================================

class TestLiteArc:
    """Tests for LiteArc low-overhead monitoring."""
    
    def test_import(self):
        """Test that LiteArc can be imported."""
        from arc.api.lite import LiteArc
        assert LiteArc is not None
    
    def test_step_skipping(self, simple_model):
        """Test that LiteArc skips non-check steps."""
        from arc.api.lite import LiteArc, LiteConfig
        
        config = LiteConfig(check_every_n_steps=10)
        arc = LiteArc(simple_model, config=config)
        
        # First 9 steps should not be checked
        for i in range(9):
            result = arc.step(torch.tensor(1.0))
            assert result["checked"] == False
        
        # 10th step should be checked
        result = arc.step(torch.tensor(1.0))
        assert result["checked"] == True
    
    def test_nan_detection(self, simple_model):
        """Test NaN loss detection in LiteArc."""
        from arc.api.lite import LiteArc, LiteConfig
        
        config = LiteConfig(check_every_n_steps=1)  # Check every step
        arc = LiteArc(simple_model, config=config)
        
        result = arc.step(torch.tensor(float('nan')))
        assert result["alert"] == "nan_loss"


# =============================================================================
# ArcV2 Integration Tests
# =============================================================================

class TestArcV2:
    """Tests for main ArcV2 API."""
    
    def test_import(self):
        """Test ArcV2 import."""
        from arc import ArcV2
        assert ArcV2 is not None
    
    def test_auto_creation(self, simple_model):
        """Test ArcV2.auto() factory."""
        from arc import ArcV2
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        arc = ArcV2.auto(simple_model, optimizer)
        
        assert arc is not None
    
    def test_step(self, simple_model):
        """Test ArcV2 step."""
        from arc import ArcV2
        
        optimizer = torch.optim.Adam(simple_model.parameters())
        arc = ArcV2.auto(simple_model, optimizer)
        
        result = arc.step(torch.tensor(1.0))
        assert "step" in result


# =============================================================================
# ArcV2.auto() Large-Model Regression Tests  (Issue #1)
# =============================================================================

class TestArcV2AutoLargeModel:
    """
    Regression tests for issue #1:
    "[BUG] ArcV2.auto() passes raw dicts to Config instead of config dataclass instances"

    Verifies that both large-model branches in ArcV2.auto() correctly construct
    SignalConfig and FeatureConfig dataclass instances rather than raw dicts, and
    that the field name 'activation_sample_ratio' (not 'activation_sample_rate')
    is used.
    """

    @staticmethod
    def _make_model_gt10m() -> nn.Module:
        """
        Build a model with more than 10M parameters.
        Linear(1024, 3000)  => 1024 * 3000 + 3000 = 3,075,000
        Linear(3000, 3000)  => 3000 * 3000 + 3000 = 9,003,000
        Linear(3000, 10)    =>    3000 * 10 + 10  =    30,010
        Total               => ~12,108,010  > 10M
        """
        return nn.Sequential(
            nn.Linear(1024, 3000),
            nn.ReLU(),
            nn.Linear(3000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 10),
        )

    def test_large_model_param_count(self):
        """Sanity check: confirm helper model exceeds 10M parameters."""
        model = self._make_model_gt10m()
        n = sum(p.numel() for p in model.parameters())
        assert n > 10_000_000, f"Expected >10M params, got {n}"

    def test_auto_no_crash_gt10m(self):
        """
        ArcV2.auto() must not crash for models with >10M parameters.

        Before the fix, this raised:
            AttributeError: 'dict' object has no attribute 'compute_curvature_proxy'
        inside Arc._init_collectors() because config.signal was a raw dict.
        """
        from arc import ArcV2

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)   # must not raise
        assert arc is not None

    def test_signal_is_SignalConfig_instance_gt10m(self):
        """config.signal must be a SignalConfig instance, not a raw dict."""
        from arc import ArcV2
        from arc.config import SignalConfig

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        assert isinstance(arc.config.signal, SignalConfig), (
            f"Expected SignalConfig, got {type(arc.config.signal).__name__}"
        )

    def test_feature_is_FeatureConfig_instance_gt10m(self):
        """config.feature must be a FeatureConfig instance, not a raw dict."""
        from arc import ArcV2
        from arc.config import FeatureConfig

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        assert isinstance(arc.config.feature, FeatureConfig), (
            f"Expected FeatureConfig, got {type(arc.config.feature).__name__}"
        )

    def test_activation_sample_ratio_is_valid_float_gt10m(self):
        """
        config.signal.activation_sample_ratio must be a valid float in (0, 1].

        Before the fix, 'activation_sample_rate' (wrong name) was stored in a
        raw dict — accessing .activation_sample_ratio would have raised
        AttributeError or returned the dataclass default instead of 0.1.
        """
        from arc import ArcV2

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        ratio = arc.config.signal.activation_sample_ratio
        assert isinstance(ratio, float), (
            f"Expected float, got {type(ratio).__name__}"
        )
        assert 0.0 < ratio <= 1.0, (
            f"activation_sample_ratio out of range: {ratio}"
        )

    def test_activation_sample_ratio_value_gt10m(self):
        """config.signal.activation_sample_ratio must equal 0.1 for >10M models."""
        from arc import ArcV2

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        assert arc.config.signal.activation_sample_ratio == 0.1

    def test_window_size_gt10m(self):
        """config.feature.window_size must equal 10 for >10M-param models."""
        from arc import ArcV2

        model = self._make_model_gt10m()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        assert arc.config.feature.window_size == 10


# =============================================================================
# Config Numeric Validation Tests  (Issue #2)
# =============================================================================

class TestConfigValidation:
    """
    Tests for __post_init__ numeric validation added to config dataclasses.

    Covers:
      - SignalConfig: activation_sample_ratio, collect_every_n_steps
      - FeatureConfig: window_size, min_history
      - PredictionConfig: dropout_rate, confidence_level, mc_dropout_samples
      - FailureThresholds: loss_explosion_factor, numeric threshold fields
      - OverheadConfig: max_overhead_percent, overhead_check_interval
      - Config.from_dict() propagates ValueError for invalid nested values
      - Default Config(), Config.low_overhead(), Config.high_accuracy() still pass
    """

    # ------------------------------------------------------------------
    # SignalConfig
    # ------------------------------------------------------------------

    def test_signal_ratio_zero_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="activation_sample_ratio"):
            SignalConfig(activation_sample_ratio=0.0)

    def test_signal_ratio_negative_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="activation_sample_ratio"):
            SignalConfig(activation_sample_ratio=-0.1)

    def test_signal_ratio_above_one_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="activation_sample_ratio"):
            SignalConfig(activation_sample_ratio=1.5)

    def test_signal_ratio_exactly_one_valid(self):
        """Boundary: ratio == 1.0 must be accepted (spec is (0, 1])."""
        from arc.config import SignalConfig
        cfg = SignalConfig(activation_sample_ratio=1.0)
        assert cfg.activation_sample_ratio == 1.0

    def test_signal_collect_steps_zero_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="collect_every_n_steps"):
            SignalConfig(collect_every_n_steps=0)

    def test_signal_collect_steps_negative_raises(self):
        from arc.config import SignalConfig
        with pytest.raises(ValueError, match="collect_every_n_steps"):
            SignalConfig(collect_every_n_steps=-1)

    # ------------------------------------------------------------------
    # FeatureConfig
    # ------------------------------------------------------------------

    def test_feature_window_size_zero_raises(self):
        from arc.config import FeatureConfig
        with pytest.raises(ValueError, match="window_size"):
            FeatureConfig(window_size=0)

    def test_feature_window_size_negative_raises(self):
        from arc.config import FeatureConfig
        with pytest.raises(ValueError, match="window_size"):
            FeatureConfig(window_size=-1)

    def test_feature_min_history_zero_raises(self):
        from arc.config import FeatureConfig
        with pytest.raises(ValueError, match="min_history"):
            FeatureConfig(min_history=0)

    def test_feature_min_history_negative_raises(self):
        from arc.config import FeatureConfig
        with pytest.raises(ValueError, match="min_history"):
            FeatureConfig(min_history=-5)

    # ------------------------------------------------------------------
    # PredictionConfig
    # ------------------------------------------------------------------

    def test_prediction_dropout_negative_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="dropout_rate"):
            PredictionConfig(dropout_rate=-0.1)

    def test_prediction_dropout_above_one_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="dropout_rate"):
            PredictionConfig(dropout_rate=1.5)

    def test_prediction_dropout_zero_valid(self):
        """Boundary: dropout_rate == 0.0 must be accepted (spec is [0, 1])."""
        from arc.config import PredictionConfig
        cfg = PredictionConfig(dropout_rate=0.0)
        assert cfg.dropout_rate == 0.0

    def test_prediction_confidence_zero_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="confidence_level"):
            PredictionConfig(confidence_level=0.0)

    def test_prediction_confidence_above_one_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="confidence_level"):
            PredictionConfig(confidence_level=1.1)

    def test_prediction_confidence_exactly_one_valid(self):
        """Boundary: confidence_level == 1.0 must be accepted (spec is (0, 1])."""
        from arc.config import PredictionConfig
        cfg = PredictionConfig(confidence_level=1.0)
        assert cfg.confidence_level == 1.0

    def test_prediction_mc_samples_zero_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="mc_dropout_samples"):
            PredictionConfig(mc_dropout_samples=0)

    def test_prediction_mc_samples_negative_raises(self):
        from arc.config import PredictionConfig
        with pytest.raises(ValueError, match="mc_dropout_samples"):
            PredictionConfig(mc_dropout_samples=-1)

    # ------------------------------------------------------------------
    # FailureThresholds
    # ------------------------------------------------------------------

    def test_thresholds_loss_factor_zero_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="loss_explosion_factor"):
            FailureThresholds(loss_explosion_factor=0.0)

    def test_thresholds_loss_factor_negative_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="loss_explosion_factor"):
            FailureThresholds(loss_explosion_factor=-1.0)

    def test_thresholds_vanishing_grad_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="vanishing_grad_threshold"):
            FailureThresholds(vanishing_grad_threshold=0.0)

    def test_thresholds_exploding_grad_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="exploding_grad_threshold"):
            FailureThresholds(exploding_grad_threshold=-1.0)

    def test_thresholds_activation_similarity_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="activation_similarity_threshold"):
            FailureThresholds(activation_similarity_threshold=0.0)

    def test_thresholds_effective_rank_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="effective_rank_collapse_ratio"):
            FailureThresholds(effective_rank_collapse_ratio=-0.1)

    def test_thresholds_overfit_gap_nonpositive_raises(self):
        from arc.config import FailureThresholds
        with pytest.raises(ValueError, match="overfit_gap_threshold"):
            FailureThresholds(overfit_gap_threshold=0.0)

    # ------------------------------------------------------------------
    # OverheadConfig
    # ------------------------------------------------------------------

    def test_overhead_percent_negative_raises(self):
        from arc.config import OverheadConfig
        with pytest.raises(ValueError, match="max_overhead_percent"):
            OverheadConfig(max_overhead_percent=-1.0)

    def test_overhead_percent_zero_valid(self):
        """Boundary: max_overhead_percent == 0.0 must be accepted (spec is >= 0)."""
        from arc.config import OverheadConfig
        cfg = OverheadConfig(max_overhead_percent=0.0)
        assert cfg.max_overhead_percent == 0.0

    def test_overhead_interval_zero_raises(self):
        from arc.config import OverheadConfig
        with pytest.raises(ValueError, match="overhead_check_interval"):
            OverheadConfig(overhead_check_interval=0)

    def test_overhead_interval_negative_raises(self):
        from arc.config import OverheadConfig
        with pytest.raises(ValueError, match="overhead_check_interval"):
            OverheadConfig(overhead_check_interval=-10)

    # ------------------------------------------------------------------
    # Config.from_dict() propagation
    # ------------------------------------------------------------------

    def test_from_dict_invalid_signal_ratio_raises(self):
        """Config.from_dict() must propagate ValueError for invalid nested values."""
        from arc.config import Config
        with pytest.raises(ValueError, match="activation_sample_ratio"):
            Config.from_dict({"signal": {"activation_sample_ratio": -1.0}})

    def test_from_dict_invalid_dropout_raises(self):
        from arc.config import Config
        with pytest.raises(ValueError, match="dropout_rate"):
            Config.from_dict({"prediction": {"dropout_rate": 2.0}})

    # ------------------------------------------------------------------
    # Preset smoke tests — must NOT raise
    # ------------------------------------------------------------------

    def test_default_config_passes(self):
        """Config() with all defaults must construct without error."""
        from arc.config import Config
        cfg = Config()
        assert cfg is not None

    def test_low_overhead_preset_passes(self):
        """Config.low_overhead() must construct without error."""
        from arc.config import Config
        cfg = Config.low_overhead()
        assert cfg is not None

    def test_high_accuracy_preset_passes(self):
        """Config.high_accuracy() must construct without error."""
        from arc.config import Config
        cfg = Config.high_accuracy()
        assert cfg is not None


# =============================================================================
# ArcV2 EWC Persistence Tests  (Issue #14)
# =============================================================================

class TestArcV2Persistence:
    """
    Regression tests for issue #14:
    "[BUG] ArcV2.save() drops EWC state silently with no load() counterpart"

    save() computed self._ewc.state_dict() but never wrote it to disk (only
    n_tasks landed in ewc_meta.json), and there was no load() to restore Fisher
    information / optimal params after a restart.
    """

    @staticmethod
    def _make_model() -> nn.Module:
        return nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    @staticmethod
    def _make_loader() -> torch.utils.data.DataLoader:
        from torch.utils.data import TensorDataset, DataLoader
        x = torch.randn(16, 100)
        y = torch.randint(0, 10, (16,))
        return DataLoader(TensorDataset(x, y), batch_size=4)

    def _consolidated_arc(self):
        from arc import ArcV2

        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters())
        arc = ArcV2(continual_learning=True, verbose=False)
        arc.attach(model, optimizer)
        arc.begin_task("task_1")
        arc.consolidate_task(self._make_loader())
        return arc

    def test_save_writes_ewc_state_file(self, tmp_path):
        """save() must persist ewc_state.pt, not just ewc_meta.json."""
        arc = self._consolidated_arc()

        arc.save(str(tmp_path))

        assert (tmp_path / "ewc_state.pt").exists(), "ewc_state.pt was not written"

    def test_load_restores_ewc_after_restart(self, tmp_path):
        """A fresh instance (no begin_task) must recover the consolidated task via load()."""
        from arc import ArcV2

        self._consolidated_arc().save(str(tmp_path))

        # Simulate a restart: brand-new instance, begin_task() is never called
        model = self._make_model()
        restored = ArcV2(continual_learning=True, verbose=False)
        restored.attach(model, torch.optim.Adam(model.parameters()))
        restored.load(str(tmp_path))

        assert restored._ewc is not None, "EWC was not reconstructed on load()"
        assert restored._ewc.n_tasks == 1

    def test_load_restores_fisher_values(self, tmp_path):
        """Restored Fisher matrices must match the ones that were saved."""
        from arc import ArcV2

        arc = self._consolidated_arc()
        arc.save(str(tmp_path))
        original_fisher = arc._ewc.task_memories[0].fisher_diag

        model = self._make_model()
        restored = ArcV2(continual_learning=True, verbose=False)
        restored.attach(model, torch.optim.Adam(model.parameters()))
        restored.load(str(tmp_path))

        restored_fisher = restored._ewc.task_memories[0].fisher_diag
        assert set(restored_fisher.keys()) == set(original_fisher.keys())
        for name, tensor in original_fisher.items():
            assert torch.allclose(tensor, restored_fisher[name])

    def test_load_uses_weights_only(self, tmp_path):
        """load() must call torch.load with weights_only=True on the first attempt
        (safe tensor-only deserialization, matching the repo's checkpoint pattern)."""
        from unittest.mock import patch
        from arc import ArcV2

        self._consolidated_arc().save(str(tmp_path))

        model = self._make_model()
        restored = ArcV2(continual_learning=True, verbose=False)
        restored.attach(model, torch.optim.Adam(model.parameters()))

        captured = []
        original_load = torch.load

        def spy(*args, **kwargs):
            captured.append(dict(kwargs))
            return original_load(*args, **kwargs)

        with patch("arc.api.v2.torch.load", side_effect=spy):
            restored.load(str(tmp_path))

        assert captured, "load() never called torch.load"
        assert captured[0].get("weights_only") is True, (
            f"ArcV2.load() must use weights_only=True on the first attempt. "
            f"Got first-call kwargs: {captured[0]}"
        )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])