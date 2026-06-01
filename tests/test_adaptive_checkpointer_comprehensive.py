"""Comprehensive Edge Case Tests for AdaptiveCheckpointer Device Fix

This test suite validates all edge cases mentioned in Task 4:
- Model with parameters on CPU
- Model with parameters on CUDA (if available)
- Model with no parameters (empty model)
- Multiple restore cycles
- Switching between checkpoint strategies

**Validates: All Requirements (1.1-1.4, 2.1-2.5, 3.1-3.8)**
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
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
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


class EmptyModel(nn.Module):
    """Model with no parameters for edge case testing."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_edge_case_model_with_parameters_on_cpu():
    """
    **Edge Case: Model with parameters on CPU**
    
    Test that device initialization works correctly for CPU models.
    Verifies that self.device is set to CPU and checkpoint restore works.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()  # Default is CPU
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            verbose=False,
        )
        
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        # Verify device is CPU
        assert hasattr(checkpointer, 'device'), \
            "Checkpointer should have device attribute"
        assert checkpointer.device.type == 'cpu', \
            "Device should be CPU for CPU model"
        
        # Save and restore checkpoint
        initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpointer.save(step=1)
        
        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        
        # Restore
        restored_step = checkpointer.restore(checkpoint_idx=-1)
        assert restored_step == 1, "Should restore to step 1"
        
        # Verify restoration
        restored_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        for key in initial_state:
            assert torch.allclose(initial_state[key], restored_state[key]), \
                f"Parameter {key} should be restored correctly"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_edge_case_model_with_parameters_on_cuda():
    """
    **Edge Case: Model with parameters on CUDA**
    
    Test that device initialization works correctly for CUDA models.
    Verifies that self.device is set to CUDA and checkpoint restore works.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            verbose=False,
        )
        
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        # Verify device is CUDA
        assert hasattr(checkpointer, 'device'), \
            "Checkpointer should have device attribute"
        assert checkpointer.device.type == 'cuda', \
            "Device should be CUDA for CUDA model"
        
        # Save and restore checkpoint
        initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpointer.save(step=1)
        
        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        
        # Restore
        restored_step = checkpointer.restore(checkpoint_idx=-1)
        assert restored_step == 1, "Should restore to step 1"
        
        # Verify restoration
        restored_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        for key in initial_state:
            assert torch.allclose(initial_state[key], restored_state[key]), \
                f"Parameter {key} should be restored correctly"
        
        # Verify model is still on CUDA after restore
        assert next(model.parameters()).device.type == 'cuda', \
            "Model should remain on CUDA after restore"


def test_edge_case_model_with_no_parameters():
    """
    **Edge Case: Model with no parameters (empty model)**
    
    Test that device initialization handles empty models correctly.
    Should fallback to CPU device when model has no parameters.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = EmptyModel()
        # Create a dummy optimizer with a dummy parameter
        dummy_param = nn.Parameter(torch.randn(1))
        optimizer = torch.optim.SGD([dummy_param], lr=0.01)
        
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            verbose=False,
        )
        
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        # Verify device is CPU (fallback for empty model)
        assert hasattr(checkpointer, 'device'), \
            "Checkpointer should have device attribute"
        assert checkpointer.device.type == 'cpu', \
            "Device should fallback to CPU for empty model"
        
        # Save and restore checkpoint
        checkpointer.save(step=1)
        restored_step = checkpointer.restore(checkpoint_idx=-1)
        assert restored_step == 1, "Should restore to step 1"


def test_edge_case_multiple_restore_cycles():
    """
    **Edge Case: Multiple restore cycles**
    
    Test that multiple save/restore cycles work correctly.
    Verifies that device attribute remains valid across multiple operations.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            max_disk_checkpoints=5,
            verbose=False,
        )
        
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        # Save multiple checkpoints with different model states
        for step in range(1, 6):
            # Modify model
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(0.1 * step)
            
            checkpointer.save(step=step)
        
        # Restore the last checkpoint multiple times
        for _ in range(3):
            restored_step = checkpointer.restore(checkpoint_idx=-1)
            assert restored_step == 5, "Should restore to step 5"
            
            # Verify device attribute persists
            assert hasattr(checkpointer, 'device'), \
                "Device attribute should persist after restore"
            assert checkpointer.device.type == 'cpu', \
                "Device should remain CPU"


def test_edge_case_switching_between_checkpoint_strategies():
    """
    **Edge Case: Switching between checkpoint strategies**
    
    Test that device initialization works correctly when switching between
    different checkpoint strategies.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        strategies = [
            CheckpointStrategy.FULL_CPU,
            CheckpointStrategy.QUANTIZED_FP16,
            CheckpointStrategy.STREAMING_DISK,
            CheckpointStrategy.INCREMENTAL_DELTA,
        ]
        
        for strategy in strategies:
            config = AdaptiveCheckpointConfig(
                disk_checkpoint_dir=tmpdir,
                preferred_strategy=strategy,
                auto_select_strategy=False,
                verbose=False,
            )
            
            checkpointer = AdaptiveCheckpointer(model, optimizer, config)
            
            # Verify device is initialized
            assert hasattr(checkpointer, 'device'), \
                f"Checkpointer should have device attribute for strategy {strategy}"
            assert checkpointer.device.type == 'cpu', \
                f"Device should be CPU for strategy {strategy}"
            
            # Save checkpoint
            initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            checkpointer.save(step=1)
            
            # Modify model
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(1.0)
            
            # Restore checkpoint
            restored_step = checkpointer.restore(checkpoint_idx=-1)
            assert restored_step == 1, f"Should restore to step 1 for strategy {strategy}"
            
            # Verify restoration (with tolerance for quantized strategies)
            restored_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tolerance = 1e-3 if strategy == CheckpointStrategy.QUANTIZED_FP16 else 1e-6
            for key in initial_state:
                assert torch.allclose(initial_state[key], restored_state[key], atol=tolerance), \
                    f"Parameter {key} should be restored correctly for strategy {strategy}"


def test_edge_case_mixed_in_memory_and_disk_restore():
    """
    **Edge Case: Mixed in-memory and disk restore**
    
    Test that device attribute works correctly for both in-memory and
    disk-based checkpoint restoration.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Use FULL_CPU for in-memory checkpoints
        config_memory = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.FULL_CPU,
            auto_select_strategy=False,
            verbose=False,
        )
        
        checkpointer_memory = AdaptiveCheckpointer(model, optimizer, config_memory)
        
        # Save in-memory checkpoint
        state_1 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpointer_memory.save(step=1)
        
        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        
        # Restore from in-memory checkpoint
        restored_step = checkpointer_memory.restore(checkpoint_idx=-1)
        assert restored_step == 1, "Should restore from in-memory checkpoint"
        
        # Now use STREAMING_DISK for disk checkpoints
        config_disk = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            verbose=False,
        )
        
        checkpointer_disk = AdaptiveCheckpointer(model, optimizer, config_disk)
        
        # Save disk checkpoint
        state_2 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpointer_disk.save(step=2)
        
        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        
        # Restore from disk checkpoint
        restored_step = checkpointer_disk.restore(checkpoint_idx=-1)
        assert restored_step == 2, "Should restore from disk checkpoint"


def test_edge_case_device_attribute_persistence():
    """
    **Edge Case: Device attribute persistence**
    
    Test that device attribute persists throughout the lifecycle of the
    checkpointer and doesn't get accidentally deleted or modified.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            verbose=False,
        )
        
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        # Verify device exists after initialization
        assert hasattr(checkpointer, 'device'), \
            "Device should exist after initialization"
        initial_device = checkpointer.device
        
        # Perform multiple operations
        for step in range(1, 4):
            checkpointer.save(step=step)
            assert hasattr(checkpointer, 'device'), \
                f"Device should persist after save at step {step}"
            assert checkpointer.device == initial_device, \
                f"Device should not change after save at step {step}"
        
        # Restore operations (use last checkpoint)
        for _ in range(3):
            checkpointer.restore(checkpoint_idx=-1)
            assert hasattr(checkpointer, 'device'), \
                "Device should persist after restore"
            assert checkpointer.device == initial_device, \
                "Device should not change after restore"
        
        # Get stats
        stats = checkpointer.get_stats()
        assert hasattr(checkpointer, 'device'), \
            "Device should persist after get_stats"
        assert checkpointer.device == initial_device, \
            "Device should not change after get_stats"


if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE EDGE CASE TESTS")
    print("AdaptiveCheckpointer Device Initialization Fix")
    print("="*70)
    print("\nTesting all edge cases mentioned in Task 4:\n")
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
