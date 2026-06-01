"""Bug Condition Exploration Test for AdaptiveCheckpointer Device Initialization Bug

This test is designed to FAIL on unfixed code to confirm the bug exists.

**Bug Description:**
The AdaptiveCheckpointer class never initializes the `self.device` attribute in its
`__init__()` method. When `restore()` is called with a disk-based checkpoint, it
attempts to use `self.device` as the `map_location` parameter for `torch.load()`,
causing an AttributeError.

**Expected Behavior on UNFIXED Code:**
This test MUST FAIL with: AttributeError: 'AdaptiveCheckpointer' object has no attribute 'device'

**Expected Behavior on FIXED Code:**
This test should PASS - checkpoint restoration completes successfully without AttributeError.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5**
"""

try:
    import pytest
except ImportError:
    pytest = None

import torch
import torch.nn as nn
import tempfile
import os
from arc.checkpointing.adaptive import (
    AdaptiveCheckpointer,
    AdaptiveCheckpointConfig,
    CheckpointStrategy,
)


class SimpleModel(nn.Module):
    """Simple model for testing with parameters."""
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


def test_bug_condition_device_missing_on_disk_restore():
    """
    **Property 1: Bug Condition** - Device Attribute Missing on Disk Restore
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
    
    **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    
    Test implementation:
    1. Create AdaptiveCheckpointer instance with a model and optimizer
    2. Save a checkpoint to disk using STREAMING_DISK strategy
    3. Call restore() method to load the disk-based checkpoint
    4. Assert that self.device attribute exists and is accessible
    5. Assert that checkpoint restoration completes successfully without AttributeError
    
    **EXPECTED OUTCOME on UNFIXED code**: 
    Test FAILS with `AttributeError: 'AdaptiveCheckpointer' object has no attribute 'device'`
    
    **EXPECTED OUTCOME on FIXED code**:
    Test PASSES - checkpoint restoration succeeds without AttributeError
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup: Create model and optimizer
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Configure checkpointer to use STREAMING_DISK strategy
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            verbose=False,
        )
        
        # Create checkpointer instance
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        
        # Save a checkpoint to disk
        metadata = checkpointer.save(step=1)
        assert metadata.strategy == CheckpointStrategy.STREAMING_DISK, \
            "Checkpoint should be saved with STREAMING_DISK strategy"
        
        # Verify checkpoint file exists on disk
        checkpoint_files = [f for f in os.listdir(tmpdir) if f.startswith('checkpoint_')]
        assert len(checkpoint_files) > 0, "Checkpoint file should exist on disk"
        
        # CRITICAL TEST: Call restore() - this will fail on unfixed code
        # On unfixed code: raises AttributeError: 'AdaptiveCheckpointer' object has no attribute 'device'
        # On fixed code: should succeed without error
        try:
            restored_step = checkpointer.restore(checkpoint_idx=-1)
            
            # If we reach here, the bug is fixed
            # Verify restoration was successful
            assert restored_step == 1, "Restored step should match saved step"
            
            # Verify device attribute exists (this is what the fix adds)
            assert hasattr(checkpointer, 'device'), \
                "AdaptiveCheckpointer should have 'device' attribute after initialization"
            
            # Verify device is a torch.device instance
            assert isinstance(checkpointer.device, torch.device), \
                "self.device should be a torch.device instance"
            
            print("✓ Test PASSED: Bug is FIXED - checkpoint restoration succeeded without AttributeError")
            
        except AttributeError as e:
            # This is the expected behavior on UNFIXED code
            error_msg = str(e)
            print(f"✗ Test FAILED (as expected on unfixed code): {error_msg}")
            
            # Document the counterexample
            print("\n=== COUNTEREXAMPLE FOUND ===")
            print(f"Error Type: AttributeError")
            print(f"Error Message: {error_msg}")
            print(f"Expected Error: 'AdaptiveCheckpointer' object has no attribute 'device'")
            
            # Verify this is the specific bug we're looking for
            assert "'AdaptiveCheckpointer' object has no attribute 'device'" in error_msg or \
                   "has no attribute 'device'" in error_msg, \
                   f"Unexpected AttributeError: {error_msg}"
            
            # Re-raise to mark test as failed (confirming bug exists)
            raise


def test_bug_condition_model_with_parameters():
    """
    Test device initialization bug with a model containing parameters.
    
    **Validates: Requirements 2.1, 2.2, 2.3**
    
    This test verifies that the bug occurs regardless of model architecture,
    as long as the model has parameters.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            verbose=False,
        )
        
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        checkpointer.save(step=10)
        
        # This should fail on unfixed code with AttributeError
        try:
            restored_step = checkpointer.restore(checkpoint_idx=-1)
            assert restored_step == 10
            assert hasattr(checkpointer, 'device')
            print("✓ Model with parameters: Bug is FIXED")
        except AttributeError as e:
            print(f"✗ Model with parameters: Bug EXISTS - {e}")
            raise


def test_bug_condition_model_without_parameters():
    """
    Test device initialization bug with a model containing NO parameters.
    
    **Validates: Requirements 2.4, 2.5**
    
    This is an edge case where the model has no parameters. The fix should
    handle this by falling back to CPU device.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model = EmptyModel()
        # Create a dummy optimizer (even though model has no parameters)
        dummy_param = nn.Parameter(torch.randn(1))
        optimizer = torch.optim.SGD([dummy_param], lr=0.01)
        
        config = AdaptiveCheckpointConfig(
            disk_checkpoint_dir=tmpdir,
            preferred_strategy=CheckpointStrategy.STREAMING_DISK,
            auto_select_strategy=False,
            verbose=False,
        )
        
        checkpointer = AdaptiveCheckpointer(model, optimizer, config)
        checkpointer.save(step=5)
        
        # This should fail on unfixed code with AttributeError
        # On fixed code, should fallback to CPU device
        try:
            restored_step = checkpointer.restore(checkpoint_idx=-1)
            assert restored_step == 5
            assert hasattr(checkpointer, 'device')
            # For empty model, device should be CPU (fallback)
            assert checkpointer.device.type == 'cpu', \
                "Empty model should fallback to CPU device"
            print("✓ Model without parameters: Bug is FIXED (CPU fallback works)")
        except AttributeError as e:
            print(f"✗ Model without parameters: Bug EXISTS - {e}")
            raise


def test_bug_condition_cuda_model():
    """
    Test device initialization bug with a CUDA model.
    
    **Validates: Requirements 2.1, 2.2, 2.3**
    
    This test verifies the bug occurs with GPU models and that the fix
    correctly extracts the CUDA device.
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
        checkpointer.save(step=20)
        
        # This should fail on unfixed code with AttributeError
        try:
            restored_step = checkpointer.restore(checkpoint_idx=-1)
            assert restored_step == 20
            assert hasattr(checkpointer, 'device')
            assert checkpointer.device.type == 'cuda', \
                "CUDA model should have CUDA device"
            print("✓ CUDA model: Bug is FIXED")
        except AttributeError as e:
            print(f"✗ CUDA model: Bug EXISTS - {e}")
            raise


def test_bug_condition_multiple_restore_cycles():
    """
    Test device initialization bug across multiple save/restore cycles.
    
    **Validates: Requirements 2.1, 2.2, 2.3**
    
    This test verifies that the bug affects all restore operations,
    not just the first one.
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
        
        # Save multiple checkpoints
        for step in [1, 2, 3]:
            checkpointer.save(step=step)
        
        # Try to restore each checkpoint - all should fail on unfixed code
        try:
            for idx in [-3, -2, -1]:
                restored_step = checkpointer.restore(checkpoint_idx=idx)
                assert restored_step in [1, 2, 3]
            
            assert hasattr(checkpointer, 'device')
            print("✓ Multiple restore cycles: Bug is FIXED")
        except AttributeError as e:
            print(f"✗ Multiple restore cycles: Bug EXISTS - {e}")
            raise


if __name__ == "__main__":
    print("="*70)
    print("BUG CONDITION EXPLORATION TEST")
    print("AdaptiveCheckpointer Device Initialization Bug")
    print("="*70)
    print("\nThis test is designed to FAIL on unfixed code.")
    print("Failure confirms the bug exists and helps understand the root cause.\n")
    
    # Run tests manually for detailed output
    tests = [
        ("Basic disk restore", test_bug_condition_device_missing_on_disk_restore),
        ("Model with parameters", test_bug_condition_model_with_parameters),
        ("Model without parameters", test_bug_condition_model_without_parameters),
        ("Multiple restore cycles", test_bug_condition_multiple_restore_cycles),
    ]
    
    if torch.cuda.is_available():
        tests.append(("CUDA model", test_bug_condition_cuda_model))
    
    results = []
    for name, test_fn in tests:
        print(f"\n--- Running: {name} ---")
        try:
            test_fn()
            results.append((name, "PASS (Bug is FIXED)"))
        except AttributeError as e:
            results.append((name, f"FAIL (Bug EXISTS): {str(e)[:50]}..."))
        except Exception as e:
            results.append((name, f"ERROR: {str(e)[:50]}..."))
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    for name, status in results:
        symbol = "✓" if "PASS" in status else "✗"
        print(f"{symbol} {name}: {status}")
    
    print("\n" + "="*70)
