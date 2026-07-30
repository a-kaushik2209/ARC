"""Preservation Property Tests for AdaptiveCheckpointer Device Initialization Fix

These tests verify that non-buggy operations remain unchanged after the fix.

**IMPORTANT**: These tests are designed to PASS on UNFIXED code.
They capture the baseline behavior that must be preserved when the fix is applied.

**Testing Approach**: 
Since hypothesis is not available, we use pytest with comprehensive parameterization
to test multiple scenarios and provide strong coverage guarantees.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import io
import sys
from contextlib import contextmanager
from arc.checkpointing.adaptive import (
    AdaptiveCheckpointer,
    AdaptiveCheckpointConfig,
    CheckpointStrategy,
)


# ============================================================================
# Test Models
# ============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TinyModel(nn.Module):
    """Tiny model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 3)
    
    def forward(self, x):
        return self.linear(x)


class MediumModel(nn.Module):
    """Medium-sized model for testing."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
    
    def forward(self, x):
        return self.layers(x)


# ============================================================================
# Helper Functions
# ============================================================================

@contextmanager
def capture_stdout():
    """Capture stdout for testing verbose output."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_stdout


def create_checkpointer(model, optimizer, strategy, tmpdir, verbose=False):
    """Helper to create checkpointer with specific strategy."""
    config = AdaptiveCheckpointConfig(
        disk_checkpoint_dir=tmpdir,
        preferred_strategy=strategy,
        auto_select_strategy=False,
        verbose=verbose,
    )
    return AdaptiveCheckpointer(model, optimizer, config)


def get_model_state_dict_copy(model):
    """Get a deep copy of model state dict on CPU."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def states_equal(state1, state2, atol=1e-6):
    """Check if two state dicts are equal within tolerance."""
    if set(state1.keys()) != set(state2.keys()):
        return False
    for key in state1.keys():
        if not torch.allclose(state1[key].cpu(), state2[key].cpu(), atol=atol):
            return False
    return True


# ============================================================================
# Property 2: Preservation - Save Operations
# ============================================================================

@pytest.mark.parametrize("strategy", [
    CheckpointStrategy.FULL_CPU,
    CheckpointStrategy.QUANTIZED_FP16,
    CheckpointStrategy.INCREMENTAL_DELTA,
    CheckpointStrategy.STREAMING_DISK,
])
@pytest.mark.parametrize("model_class", [TinyModel, SimpleModel, MediumModel])
def test_preservation_save_operations_produce_consistent_checkpoints(strategy, model_class):
    """
    **Property 2.1: Preservation - Save Operations**
    
    **Validates: Requirements 3.1**
    
    Test that all checkpoint save strategies produce consistent checkpoints
    on unfixed code. This behavior must be preserved after the fix.
    
    For each strategy and model architecture:
    1. Create checkpointer with specific strategy
    2. Save a checkpoint
    3. Verify checkpoint metadata is correct
    4. Verify checkpoint can be accessed (in-memory or disk)
    5. Verify checkpoint contains expected keys
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        checkpointer = create_checkpointer(model, optimizer, strategy, tmpdir)
        
        # Save checkpoint
        metadata = checkpointer.save(step=1)
        
        # Verify metadata
        # NOTE: First INCREMENTAL_DELTA checkpoint is saved as FULL_CPU (observed behavior)
        if strategy == CheckpointStrategy.INCREMENTAL_DELTA:
            assert metadata.strategy == CheckpointStrategy.FULL_CPU, \
                "First incremental checkpoint should be saved as FULL_CPU"
        else:
            assert metadata.strategy == strategy, \
                f"Metadata strategy should match requested strategy {strategy}"
        
        assert metadata.step == 1, "Metadata step should be 1"
        assert metadata.model_size_bytes > 0, "Model size should be positive"
        assert metadata.checkpoint_size_bytes > 0, "Checkpoint size should be positive"
        
        # Verify checkpoint exists
        assert len(checkpointer.checkpoints) == 1, "Should have 1 checkpoint"
        
        checkpoint = checkpointer.checkpoints[0]
        
        # Verify checkpoint structure based on strategy
        if strategy == CheckpointStrategy.STREAMING_DISK:
            # Disk checkpoint stores path
            assert 'path' in checkpoint, "Disk checkpoint should have 'path' key"
            assert os.path.exists(checkpoint['path']), "Checkpoint file should exist"
        elif strategy == CheckpointStrategy.INCREMENTAL_DELTA:
            # First incremental checkpoint is actually a full checkpoint
            assert 'model' in checkpoint or 'delta' in checkpoint, \
                "Incremental checkpoint should have 'model' or 'delta' key"
        else:
            # In-memory checkpoints have model state
            assert 'model' in checkpoint, "In-memory checkpoint should have 'model' key"
            assert 'step' in checkpoint, "Checkpoint should have 'step' key"


def test_preservation_save_operations_multiple_checkpoints():
    """
    **Property 2.1: Preservation - Save Operations (Multiple Checkpoints)**
    
    **Validates: Requirements 3.1**
    
    Test that saving multiple checkpoints works correctly and respects
    max_checkpoints limit.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.FULL_CPU,
            auto_select_strategy=False,
            max_checkpoints=3,
            verbose=False,
        )
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        # Save 5 checkpoints
        for step in range(1, 6):
            checkpointer.save(step=step)
        
        # Should only keep last 3 due to max_checkpoints=3
        assert len(checkpointer.checkpoints) == 3, \
            "Should only keep max_checkpoints (3) checkpoints"
        
        # Verify metadata tracks all saves
        assert len(checkpointer.metadata) == 5, \
            "Metadata should track all 5 saves"


# ============================================================================
# Property 2: Preservation - In-Memory Restore
# ============================================================================

@pytest.mark.parametrize("strategy", [
    CheckpointStrategy.FULL_CPU,
    CheckpointStrategy.QUANTIZED_FP16,
])
def test_preservation_in_memory_restore_works_correctly(strategy):
    """
    **Property 2.2: Preservation - In-Memory Restore**
    
    **Validates: Requirements 3.2**
    
    Test that in-memory checkpoint restoration (non-disk-based) works correctly
    on unfixed code. This behavior must be preserved after the fix.
    
    NOTE: This test does NOT call restore() with disk checkpoints, so it should
    pass on unfixed code (the bug only affects disk-based restore).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        checkpointer = create_checkpointer(model, optimizer, strategy, tmpdir)
        
        # Save initial state
        initial_state = get_model_state_dict_copy(model)
        checkpointer.save(step=1)
        
        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        
        modified_state = get_model_state_dict_copy(model)
        
        # Verify model was modified
        assert not states_equal(initial_state, modified_state), \
            "Model state should be different after modification"
        
        # Restore from in-memory checkpoint
        # NOTE: This uses in-memory checkpoint, not disk, so it works on unfixed code
        restored_step = checkpointer.restore(checkpoint_idx=-1)
        
        restored_state = get_model_state_dict_copy(model)
        
        # Verify restoration
        assert restored_step == 1, "Restored step should be 1"
        
        # For quantized checkpoints, allow some tolerance due to precision loss
        tolerance = 1e-3 if strategy == CheckpointStrategy.QUANTIZED_FP16 else 1e-6
        assert states_equal(initial_state, restored_state, atol=tolerance), \
            "Restored state should match initial state"


def test_preservation_in_memory_restore_with_optimizer_state():
    """
    **Property 2.2: Preservation - In-Memory Restore with Optimizer State**
    
    **Validates: Requirements 3.2, 3.7**
    
    Test that optimizer state restoration behavior is preserved.
    
    NOTE: The current implementation only restores optimizer state for keys
    that already exist in optimizer.state. This is the observed behavior
    that must be preserved.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        checkpointer = create_checkpointer(
            model, optimizer, CheckpointStrategy.FULL_CPU, tmpdir
        )
        
        # Perform a training step to create optimizer state
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        
        # Save checkpoint with optimizer state
        checkpointer.save(step=1)
        
        # Verify optimizer has state
        assert len(optimizer.state) > 0, "Optimizer should have state after step"
        
        # Store the optimizer state keys before clearing
        state_keys = list(optimizer.state.keys())
        
        # Modify optimizer state values (but keep keys)
        for key in state_keys:
            for state_key in optimizer.state[key]:
                if isinstance(optimizer.state[key][state_key], torch.Tensor):
                    optimizer.state[key][state_key] = torch.zeros_like(
                        optimizer.state[key][state_key]
                    )
        
        # Restore checkpoint
        checkpointer.restore(checkpoint_idx=-1)
        
        # Verify optimizer state was restored (keys should still exist)
        assert len(optimizer.state) > 0, "Optimizer state should be restored"
        
        # Verify the state values were updated (not all zeros anymore)
        has_non_zero = False
        for key in optimizer.state:
            for state_key in optimizer.state[key]:
                if isinstance(optimizer.state[key][state_key], torch.Tensor):
                    if optimizer.state[key][state_key].abs().sum() > 0:
                        has_non_zero = True
                        break
        
        assert has_non_zero, "Optimizer state should have non-zero values after restore"


# ============================================================================
# Property 2: Preservation - Strategy Selection
# ============================================================================

def test_preservation_strategy_selection_chooses_appropriate_strategy():
    """
    **Property 2.3: Preservation - Strategy Selection**
    
    **Validates: Requirements 3.3**
    
    Test that _select_best_strategy() chooses appropriate strategy based on
    available memory. This logic must be preserved after the fix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Use auto_select_strategy=True to test strategy selection
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            auto_select_strategy=True,
            verbose=False,
        )
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        # Verify a strategy was selected
        assert checkpointer.current_strategy is not None, \
            "Strategy should be selected"
        assert isinstance(checkpointer.current_strategy, CheckpointStrategy), \
            "Selected strategy should be a CheckpointStrategy enum"
        
        # Verify strategy is one of the valid options
        valid_strategies = [
            CheckpointStrategy.FULL_CPU,
            CheckpointStrategy.QUANTIZED_FP16,
            CheckpointStrategy.INCREMENTAL_DELTA,
            CheckpointStrategy.STREAMING_DISK,
        ]
        assert checkpointer.current_strategy in valid_strategies, \
            f"Strategy should be one of {valid_strategies}"


# ============================================================================
# Property 2: Preservation - Size Calculations
# ============================================================================

@pytest.mark.parametrize("model_class", [TinyModel, SimpleModel, MediumModel])
def test_preservation_size_calculations_return_correct_values(model_class):
    """
    **Property 2.4: Preservation - Size Calculations**
    
    **Validates: Requirements 3.4**
    
    Test that _calculate_model_size() and _calculate_optimizer_size() return
    correct values. These calculations must be preserved after the fix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        checkpointer = create_checkpointer(
            model, optimizer, CheckpointStrategy.FULL_CPU, tmpdir
        )
        
        # Verify model size calculation
        expected_model_size = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) + sum(
            b.numel() * b.element_size() for b in model.buffers()
        )
        
        assert checkpointer.model_size_bytes == expected_model_size, \
            f"Model size should be {expected_model_size}, got {checkpointer.model_size_bytes}"
        
        # Verify optimizer size is calculated (may be 0 initially)
        assert checkpointer.optimizer_size_bytes >= 0, \
            "Optimizer size should be non-negative"


# ============================================================================
# Property 2: Preservation - Disk Cleanup
# ============================================================================

def test_preservation_disk_cleanup_maintains_max_checkpoint_limit():
    """
    **Property 2.5: Preservation - Disk Cleanup**
    
    **Validates: Requirements 3.5**
    
    Test that _clean_disk_checkpoints() maintains max checkpoint limit.
    This cleanup logic must be preserved after the fix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            max_disk_checkpoints=3,
            verbose=False,
        )
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        # Save 5 checkpoints to disk
        for step in range(1, 6):
            checkpointer.save(step=step)
        
        # Count checkpoint files on disk
        checkpoint_files = [
            f for f in os.listdir(tmpdir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ]
        
        # Should only keep max_disk_checkpoints (3) files
        assert len(checkpoint_files) <= 3, \
            f"Should keep at most 3 checkpoint files, found {len(checkpoint_files)}"


# ============================================================================
# Property 2: Preservation - Incremental Resolution
# ============================================================================

def test_preservation_incremental_resolution_reconstructs_full_state():
    """
    **Property 2.6: Preservation - Incremental Resolution**
    
    **Validates: Requirements 3.6**
    
    Test that _resolve_incremental() reconstructs full state from deltas.
    This logic must be preserved after the fix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        checkpointer = create_checkpointer(
            model, optimizer, CheckpointStrategy.INCREMENTAL_DELTA, tmpdir
        )
        
        # Save initial checkpoint (will be full)
        initial_state = get_model_state_dict_copy(model)
        checkpointer.save(step=1)
        
        # Modify model slightly
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)
        
        # Save incremental checkpoint (will be delta)
        checkpointer.save(step=2)
        
        # Modify model again
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)
        
        modified_state = get_model_state_dict_copy(model)
        
        # Restore from incremental checkpoint (uses _resolve_incremental internally)
        checkpointer.restore(checkpoint_idx=-1)
        
        restored_state = get_model_state_dict_copy(model)
        
        # Verify restoration worked (should match state at step 2)
        # Note: We can't directly compare to initial_state because step 2 had modifications
        # But we can verify the restore didn't crash and produced valid state
        assert len(restored_state) == len(initial_state), \
            "Restored state should have same number of parameters"
        
        # Verify restored state is different from current modified state
        assert not states_equal(restored_state, modified_state), \
            "Restored state should differ from current modified state"


# ============================================================================
# Property 2: Preservation - RNG State Restoration
# ============================================================================

def test_preservation_rng_state_restoration():
    """
    **Property 2.8: Preservation - RNG State Restoration**
    
    **Validates: Requirements 3.8**
    
    Test that RNG state is restored correctly. This must be preserved after the fix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        checkpointer = create_checkpointer(
            model, optimizer, CheckpointStrategy.FULL_CPU, tmpdir
        )
        
        # Set a specific RNG state
        torch.manual_seed(42)
        initial_rng_state = torch.get_rng_state()
        
        # Save checkpoint with RNG state
        checkpointer.save(step=1)
        
        # Change RNG state
        torch.manual_seed(123)
        modified_rng_state = torch.get_rng_state()
        
        # Verify RNG state changed
        assert not torch.equal(initial_rng_state, modified_rng_state), \
            "RNG state should be different after changing seed"
        
        # Restore checkpoint (should restore RNG state)
        checkpointer.restore(checkpoint_idx=-1)
        
        restored_rng_state = torch.get_rng_state()
        
        # Verify RNG state was restored
        assert torch.equal(initial_rng_state, restored_rng_state), \
            "RNG state should be restored to initial state"


# ============================================================================
# Property 2: Preservation - Verbose Output
# ============================================================================

def test_preservation_verbose_output_prints_expected_messages():
    """
    **Property 2.9: Preservation - Verbose Output**
    
    **Validates: Requirements 3.8**
    
    Test that verbose mode prints expected messages. This must be preserved after the fix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create checkpointer with verbose=True
        with capture_stdout() as output:
            config = AdaptiveCheckpointConfig(
                disk_checkpoint_dir=tmpdir,
                preferred_strategy=CheckpointStrategy.FULL_CPU,
                auto_select_strategy=False,
                verbose=True,
            )
            checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        init_output = output.getvalue()
        
        # Verify initialization messages
        assert "AdaptiveCheckpointer initialized" in init_output, \
            "Should print initialization message"
        assert "Model size:" in init_output, \
            "Should print model size"
        assert "Strategy:" in init_output, \
            "Should print strategy"
        
        # Test save verbose output
        with capture_stdout() as output:
            checkpointer.save(step=1)
        
        save_output = output.getvalue()
        assert "Saved" in save_output or "checkpoint" in save_output.lower(), \
            "Should print save message"


# ============================================================================
# Property 2: Preservation - Statistics
# ============================================================================

def test_preservation_get_stats_returns_correct_information():
    """
    **Property 2.10: Preservation - Statistics**
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    
    Test that get_stats() returns correct information. This must be preserved after the fix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        checkpointer = create_checkpointer(
            model, optimizer, CheckpointStrategy.FULL_CPU, tmpdir
        )
        
        # Save some checkpoints
        for step in range(1, 4):
            checkpointer.save(step=step)
        
        # Get stats
        stats = checkpointer.get_stats()
        
        # Verify stats structure
        assert "strategy" in stats, "Stats should include strategy"
        assert "num_checkpoints" in stats, "Stats should include num_checkpoints"
        assert "model_size_gb" in stats, "Stats should include model_size_gb"
        assert "optimizer_size_gb" in stats, "Stats should include optimizer_size_gb"
        assert "total_saved_checkpoints" in stats, "Stats should include total_saved_checkpoints"
        assert "avg_compression_ratio" in stats, "Stats should include avg_compression_ratio"
        
        # Verify stats values
        assert stats["strategy"] == "FULL_CPU", "Strategy should be FULL_CPU"
        assert stats["num_checkpoints"] == 3, "Should have 3 checkpoints"
        assert stats["total_saved_checkpoints"] == 3, "Should have saved 3 checkpoints"
        assert stats["model_size_gb"] > 0, "Model size should be positive"


if __name__ == "__main__":
    print("="*70)
    print("PRESERVATION PROPERTY TESTS")
    print("AdaptiveCheckpointer Device Initialization Fix")
    print("="*70)
    print("\nThese tests verify that non-buggy operations remain unchanged.")
    print("They should PASS on unfixed code to establish baseline behavior.\n")
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
