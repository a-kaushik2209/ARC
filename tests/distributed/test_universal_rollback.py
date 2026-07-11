import torch
import torch.nn as nn
from torch.optim import Adam
from arc.distributed.universal import UniversalDistributedRollback

def test_restore_checkpoint_clears_warmed_optimizer_state():
    """
    Step-0 checkpoint (empty optimizer state) restored into a warmed
    Adam optimizer should not retain moment buffers from step 100.
    """
    model = nn.Linear(4, 4)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Simulate saving checkpoint at step 0 (before any optimizer.step())
    checkpoint = {
        'step': 0,
        'model': {k: v.clone() for k, v in model.state_dict().items()},
        'optimizer': {},
        'optimizer_param_groups': optimizer.state_dict()['param_groups'],
        'rng': {
            'torch': torch.get_rng_state(),
        },
    }

    # Train for 100 steps to warm up the optimizer
    for _ in range(100):
        optimizer.zero_grad()
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()

    # Confirm optimizer state is warmed
    assert len(optimizer.state) > 0, "Optimizer should have state after 100 steps"

    # Now trigger rollback to step-0 checkpoint
    rollback = UniversalDistributedRollback(model, optimizer)
    rollback._checkpoints.append(checkpoint)
    rollback._restore_checkpoint(-1)

    # After restore, optimizer state should be empty (matches step-0 checkpoint)
    assert len(optimizer.state) == 0, \
        "Optimizer state should be empty after restoring step-0 checkpoint"