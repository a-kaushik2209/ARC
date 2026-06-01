# Implementation Plan

## Phase 1: Exploratory Bug Condition Testing (BEFORE Fix)

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Device Attribute Missing on Disk Restore
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: For deterministic bugs, scope the property to the concrete failing case(s) to ensure reproducibility
  - Test implementation details from Bug Condition in design:
    - Create AdaptiveCheckpointer instance with a model and optimizer
    - Save a checkpoint to disk using STREAMING_DISK strategy
    - Call `restore()` method to load the disk-based checkpoint
    - Assert that `self.device` attribute exists and is accessible
    - Assert that checkpoint restoration completes successfully without AttributeError
  - The test assertions should match the Expected Behavior Properties from design (requirements 2.1-2.5)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS with `AttributeError: 'AdaptiveCheckpointer' object has no attribute 'device'` (this is correct - it proves the bug exists)
  - Document counterexamples found to understand root cause:
    - Which line in `restore()` raises the AttributeError
    - What the exact error message is
    - Whether the error occurs during `torch.load()` call
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5_

## Phase 2: Preservation Property Testing (BEFORE Fix)

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Non-Restore Operations Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs (operations that don't involve disk-based restore)
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements
  - Property-based testing generates many test cases for stronger guarantees
  - Test cases to observe and capture:
    1. **Save Operations**: All checkpoint save strategies (FULL_CPU, QUANTIZED_FP16, QUANTIZED_INT8, INCREMENTAL_DELTA, STREAMING_DISK) produce consistent checkpoints
    2. **In-Memory Restore**: In-memory checkpoint restoration (non-disk-based) works correctly
    3. **Strategy Selection**: `_select_best_strategy()` chooses appropriate strategy based on memory
    4. **Size Calculations**: `_calculate_model_size()` and `_calculate_optimizer_size()` return correct values
    5. **Disk Cleanup**: `_clean_disk_checkpoints()` maintains max checkpoint limit
    6. **Incremental Resolution**: `_resolve_incremental()` reconstructs full state from deltas
    7. **Optimizer State Restoration**: Optimizer state is restored to correct device
    8. **RNG State Restoration**: RNG state is restored correctly
    9. **Verbose Output**: Verbose mode prints expected messages
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

## Phase 3: Implementation

- [x] 3. Fix for missing device attribute initialization

  - [x] 3.1 Implement the device initialization fix
    - Open `arc/checkpointing/adaptive.py`
    - Locate the `__init__()` method (lines 95-125)
    - After line 103 (`self.optimizer = optimizer`), add device initialization logic:
      ```python
      # Initialize device from model parameters with CPU fallback
      try:
          self.device = next(self.model.parameters()).device
      except StopIteration:
          # Model has no parameters, fallback to CPU
          self.device = torch.device('cpu')
      ```
    - Ensure the code is inserted immediately after storing model and optimizer references
    - Verify the try-except handles the edge case of models with no parameters
    - _Bug_Condition: isBugCondition(input) where input.method == 'restore' AND input.checkpoint_type == 'disk-based' AND NOT hasattr(input.checkpointer_instance, 'device')_
    - _Expected_Behavior: self.device is initialized during construction; restore() can successfully use self.device as map_location parameter without AttributeError_
    - _Preservation: All checkpoint save/restore functionality, strategy selection, memory management, and state handling remain unchanged_
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Device Attribute Initialized and Accessible
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - Verify that:
      - `self.device` attribute exists after AdaptiveCheckpointer instantiation
      - `restore()` method completes successfully without AttributeError
      - Checkpoint is loaded correctly from disk
      - Model state is restored to the correct device
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Non-Restore Operations Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions):
      - Save operations produce identical checkpoints
      - In-memory restore works identically
      - Strategy selection chooses same strategies
      - Size calculations return same values
      - Disk cleanup maintains same behavior
      - Incremental resolution reconstructs state identically
      - Optimizer state restoration works identically
      - RNG state restoration works identically
      - Verbose output prints identical messages
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

## Phase 4: Validation

- [x] 4. Checkpoint - Ensure all tests pass
  - Run the complete test suite for AdaptiveCheckpointer
  - Verify bug condition test passes (task 1 test now succeeds)
  - Verify preservation tests pass (task 2 tests still succeed)
  - Run any existing unit tests for AdaptiveCheckpointer
  - Test edge cases:
    - Model with parameters on CPU
    - Model with parameters on CUDA (if available)
    - Model with no parameters (empty model)
    - Multiple restore cycles
    - Switching between checkpoint strategies
  - Ensure all tests pass, ask the user if questions arise
  - _Requirements: All (1.1-1.4, 2.1-2.5, 3.1-3.8)_
