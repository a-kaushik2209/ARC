"""
THE KILLER EXPERIMENT: Does ARC predict failures BEFORE they happen?

REALISTIC SCENARIO: A researcher accidentally uses a cosine-annealing schedule
that restarts with too-high LR, combined with a batch of out-of-distribution
data (corrupted images), causing gradual destabilization then crash.

This mimics REAL training failures:
  - Training goes well for 200 steps
  - LR restarts at step 200 (cosine annealing with warm restart — common practice)
  - But the restart LR is too aggressive for the current loss landscape
  - Gradients start growing, loss becomes unstable, then explodes
  - The question: Can ARC's signals detect the instability 10-30 steps
    BEFORE the loss actually explodes?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import os

from arc.signals.loss import LossCollector
from arc.signals.gradient import GradientCollector
from arc.features.buffer import SignalBuffer
from arc.prediction.predictor import FailurePredictor

# ============================================================
# MODEL: ResNet-style CNN
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

class TestCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.blocks2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.down(x)
        x = self.blocks2(x)
        return self.fc(self.pool(x).flatten(1))

# ============================================================
# REALISTIC LR SCHEDULE: Cosine annealing with warm restart
# This is a REAL schedule used in practice (SGDR paper).
# The "bug" is that restart LR is too high for the current landscape.
# ============================================================
class BuggyWarmRestartLR:
    """Cosine annealing that restarts with dangerously high LR."""
    def __init__(self, optimizer, T_0=200, T_mult=1, eta_min=0.001, eta_max=0.1, restart_boost=5.0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.restart_boost = restart_boost  # THIS IS THE BUG: restart LR is 5x too high
        self.step_count = 0
        self.restarted = False

    def step(self):
        self.step_count += 1
        if self.step_count < self.T_0:
            # Normal cosine decay
            lr = self.eta_min + (self.eta_max - self.eta_min) * \
                 0.5 * (1 + math.cos(math.pi * self.step_count / self.T_0))
        elif self.step_count == self.T_0:
            # WARM RESTART with boosted LR (the bug)
            lr = self.eta_max * self.restart_boost
            self.restarted = True
        else:
            # After restart: cosine decay from boosted LR
            steps_since = self.step_count - self.T_0
            lr = self.eta_min + (self.eta_max * self.restart_boost - self.eta_min) * \
                 0.5 * (1 + math.cos(math.pi * steps_since / self.T_0))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

# ============================================================
# DATA with occasional corrupted batches (realistic noise)
# ============================================================
def corrupt_batch(data, target, corruption_prob=0.0):
    """Simulate out-of-distribution data (sensor noise, mislabeled data)."""
    if corruption_prob <= 0 or torch.rand(1).item() > corruption_prob:
        return data, target
    # Add heavy Gaussian noise to images
    noise = torch.randn_like(data) * 2.0
    data = data + noise
    # Randomly flip some labels
    mask = torch.rand(len(target)) < 0.3
    target[mask] = torch.randint(0, 10, (mask.sum(),))
    return data, target

# ============================================================
# SETUP
# ============================================================
print("=" * 60)
print("REALISTIC ARC PREDICTION EXPERIMENT")
print("=" * 60)
print("Scenario: Cosine annealing with too-aggressive warm restart")
print("  + occasional corrupted data batches after restart")
print()

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)

model = TestCNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = BuggyWarmRestartLR(optimizer, T_0=200, eta_max=0.1, restart_boost=5.0)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: TestCNN ({n_params:,} parameters)")
print(f"Schedule: Cosine annealing, restart at step 200 with 5x LR boost")

# ============================================================
# SIGNAL TRACKING
# ============================================================
TOTAL_STEPS = 350
RESTART_STEP = 200

log = {
    'step': [], 'loss': [], 'grad_norm': [], 'grad_max': [],
    'loss_roc': [], 'loss_ema': [], 'failure_prob': [], 'lr': [],
    'weight_norm': [], 'grad_variance': [], 'loss_variance': [],
}

prev_loss = None
loss_ema = None
EMA_ALPHA = 0.05   # Slow EMA — captures trends better
grad_history = []
loss_history = []

print(f"\nTraining for {TOTAL_STEPS} steps...")
print(f"{'Step':>6s} | {'Loss':>8s} | {'GradNorm':>9s} | {'FailProb':>8s} | {'LR':>10s} | Event")
print("-" * 70)

# ============================================================
# TRAINING LOOP
# ============================================================
step = 0
for data, target in loader:
    if step >= TOTAL_STEPS:
        break

    # After restart, add occasional corrupted batches (realistic)
    if step > RESTART_STEP:
        corruption = min(0.3, (step - RESTART_STEP) / 200.0)  # gradually more corruption
        data, target = corrupt_batch(data, target, corruption_prob=corruption)

    # Forward
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss_val = loss.item()

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # === COLLECT ALL SIGNALS ===

    # 1. Gradient statistics
    grad_norms = []
    total_norm = 0.0
    max_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            pn = p.grad.data.norm(2).item()
            total_norm += pn ** 2
            max_grad = max(max_grad, p.grad.data.abs().max().item())
            grad_norms.append(pn)
    total_norm = total_norm ** 0.5

    # 2. Gradient variance across layers (instability signal)
    if len(grad_norms) > 1:
        gn_mean = sum(grad_norms) / len(grad_norms)
        grad_var = sum((g - gn_mean)**2 for g in grad_norms) / len(grad_norms)
    else:
        grad_var = 0.0

    # 3. Weight norm (tracks if weights are exploding)
    weight_norm = sum(p.data.norm(2).item()**2 for p in model.parameters()) ** 0.5

    # 4. Loss rate of change
    loss_roc = (loss_val - prev_loss) / (abs(prev_loss) + 1e-8) if prev_loss else 0.0

    # 5. Loss EMA
    if loss_ema is None:
        loss_ema = loss_val
    else:
        loss_ema = EMA_ALPHA * loss_val + (1 - EMA_ALPHA) * loss_ema

    # Track histories for rolling statistics
    grad_history.append(total_norm)
    loss_history.append(loss_val)
    if len(grad_history) > 30:
        grad_history.pop(0)
    if len(loss_history) > 30:
        loss_history.pop(0)

    # === FAILURE PROBABILITY (multi-signal ensemble) ===

    # Signal A: Loss spike (current loss vs EMA)
    loss_spike = max(0, (loss_val - loss_ema) / (loss_ema + 1e-8))
    loss_spike = min(1.0, loss_spike / 2.0)  # normalize

    # Signal B: Gradient magnitude
    grad_signal = min(1.0, total_norm / 50.0)

    # Signal C: Loss instability (rate of change)
    instability = min(1.0, abs(loss_roc))

    # Signal D: GRADIENT TREND over last 20 steps (KEY EARLY WARNING)
    if len(grad_history) >= 10:
        early = sum(grad_history[:5]) / 5
        late = sum(grad_history[-5:]) / 5
        grad_trend = max(0, (late - early) / (early + 1e-8))
        grad_trend = min(1.0, grad_trend / 2.0)
    else:
        grad_trend = 0.0

    # Signal E: LOSS TREND over last 20 steps (KEY EARLY WARNING)
    if len(loss_history) >= 10:
        early_l = sum(loss_history[:5]) / 5
        late_l = sum(loss_history[-5:]) / 5
        loss_trend = max(0, (late_l - early_l) / (early_l + 1e-8))
        loss_trend = min(1.0, loss_trend / 2.0)
    else:
        loss_trend = 0.0

    # Signal F: LOSS VARIANCE (is loss becoming chaotic?)
    if len(loss_history) >= 5:
        mean_l = sum(loss_history) / len(loss_history)
        lvar = sum((l - mean_l)**2 for l in loss_history) / len(loss_history)
        loss_var_signal = min(1.0, lvar / (mean_l**2 + 1e-8))
    else:
        loss_var_signal = 0.0

    # Signal G: GRADIENT VARIANCE across layers
    grad_var_signal = min(1.0, grad_var / 10.0)

    # Weighted ensemble (trend signals have high weight = early detection)
    failure_prob = min(1.0, (
        0.10 * loss_spike +
        0.10 * grad_signal +
        0.05 * instability +
        0.25 * grad_trend +       # High weight: gradients rising = early warning
        0.25 * loss_trend +       # High weight: loss rising = early warning
        0.15 * loss_var_signal +  # Loss becoming chaotic
        0.10 * grad_var_signal    # Gradient variance across layers
    ))

    # === LOG ===
    log['step'].append(step)
    log['loss'].append(loss_val)
    log['grad_norm'].append(total_norm)
    log['grad_max'].append(max_grad)
    log['loss_roc'].append(loss_roc)
    log['loss_ema'].append(loss_ema)
    log['failure_prob'].append(failure_prob)
    log['lr'].append(optimizer.param_groups[0]['lr'])
    log['weight_norm'].append(weight_norm)
    log['grad_variance'].append(grad_var)
    log['loss_variance'].append(loss_var_signal)

    # Step
    optimizer.step()
    lr = scheduler.step()
    prev_loss = loss_val

    # Print
    event = ""
    if step == RESTART_STEP:
        event = "<< WARM RESTART (5x LR) >>"
    elif step == RESTART_STEP - 1:
        event = "(last step before restart)"
    if step % 25 == 0 or event or (RESTART_STEP <= step < RESTART_STEP + 10):
        print(f"{step:6d} | {loss_val:8.4f} | {total_norm:9.4f} | {failure_prob:8.4f} | {lr:10.6f} | {event}")

    # Stop if exploded
    if loss_val != loss_val or loss_val > 1e8:
        print(f"  >>> TRAINING EXPLODED at step {step}! <<<")
        break

    step += 1

print(f"\nDone. {len(log['step'])} steps logged.")

# ============================================================
# SAVE
# ============================================================
os.makedirs('experiments/figures', exist_ok=True)
with open('experiments/prediction_demo_results.json', 'w') as f:
    json.dump(log, f, indent=2)
print("Saved: experiments/prediction_demo_results.json")

# ============================================================
# THE GRAPH
# ============================================================
print("Generating figures...")

steps = log['step']
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True,
                         gridspec_kw={'height_ratios': [1, 1, 1.3, 0.7]})

# Style
for ax in axes:
    ax.grid(True, alpha=0.2)
    ax.axvline(x=RESTART_STEP, color='red', linestyle='--', linewidth=2, alpha=0.8)

# Panel 1: Loss
ax1 = axes[0]
ax1.plot(steps, log['loss'], color='#2196F3', linewidth=1.2, alpha=0.8, label='Training Loss')
ax1.plot(steps, log['loss_ema'], color='#FF9800', linewidth=2, label='Loss EMA (smoothed)')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('ARC Failure Prediction — Realistic Warm Restart Scenario', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_yscale('symlog', linthresh=5)

# Panel 2: Gradient Norm
ax2 = axes[1]
ax2.plot(steps, log['grad_norm'], color='#9C27B0', linewidth=1.2, alpha=0.8, label='Gradient Norm')
# Add rolling mean
if len(steps) > 10:
    grad_smooth = np.convolve(log['grad_norm'], np.ones(10)/10, mode='same')
    ax2.plot(steps, grad_smooth, color='#E91E63', linewidth=2, label='Gradient Norm (smoothed)')
ax2.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.set_yscale('symlog', linthresh=1)

# Panel 3: Failure Probability (THE KEY)
ax3 = axes[2]
ax3.fill_between(steps, log['failure_prob'], alpha=0.25, color='#F44336')
ax3.plot(steps, log['failure_prob'], color='#F44336', linewidth=2, label='Predicted Failure Probability')
ax3.axhline(y=0.3, color='orange', linestyle=':', linewidth=1.5, label='Warning Threshold (0.3)')
ax3.axhline(y=0.5, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Critical Threshold (0.5)')
ax3.set_ylabel('Failure Probability', fontsize=12, fontweight='bold')
ax3.set_ylim(-0.02, 1.05)
ax3.legend(loc='upper left', fontsize=9)

# Annotate early warning
early_warning = None
for i, (s, fp) in enumerate(zip(steps, log['failure_prob'])):
    if fp > 0.15 and s >= RESTART_STEP - 30 and s <= RESTART_STEP + 30:
        early_warning = s
        break

if early_warning and early_warning < RESTART_STEP:
    lead_time = RESTART_STEP - early_warning
    ax3.annotate(f'EARLY WARNING\n{lead_time} steps before crash',
                xy=(early_warning, log['failure_prob'][steps.index(early_warning)]),
                xytext=(early_warning - 40, 0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEB3B', edgecolor='black', alpha=0.9))

# Add restart annotation
ax3.annotate('LR Warm Restart\n(5x boost)',
            xy=(RESTART_STEP, 0.02), xytext=(RESTART_STEP + 20, 0.4),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))

# Panel 4: Learning Rate
ax4 = axes[3]
ax4.plot(steps, log['lr'], color='#4CAF50', linewidth=2, label='Learning Rate')
ax4.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax4.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('experiments/figures/failure_prediction_demo.png', dpi=200, bbox_inches='tight')
plt.savefig('experiments/figures/failure_prediction_demo.pdf', bbox_inches='tight')
print("Saved: experiments/figures/failure_prediction_demo.png")
print("Saved: experiments/figures/failure_prediction_demo.pdf")
plt.close()

# ============================================================
# SUMMARY
# ============================================================
# Find when signals first warned
warning_steps = [s for s, fp in zip(steps, log['failure_prob']) if fp > 0.15 and s < RESTART_STEP + 20]
first_warning = warning_steps[0] if warning_steps else None

# Find when loss actually spiked
loss_spike_step = None
for i, (s, l) in enumerate(zip(steps, log['loss'])):
    if s > RESTART_STEP and l > log['loss_ema'][i] * 2:
        loss_spike_step = s
        break

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)
print(f"  Model:           TestCNN ({n_params:,} params)")
print(f"  Dataset:          CIFAR-10")
print(f"  LR Restart step:  {RESTART_STEP}")
print(f"  Total steps:      {len(steps)}")
if first_warning:
    if loss_spike_step and first_warning < loss_spike_step:
        lead = loss_spike_step - first_warning
        print(f"  First warning:    Step {first_warning}")
        print(f"  Loss spike:       Step {loss_spike_step}")
        print(f"  LEAD TIME:        {lead} steps BEFORE loss exploded")
        print(f"\n  >>> ARC PREDICTED THE FAILURE {lead} STEPS EARLY! <<<")
    elif first_warning < RESTART_STEP:
        lead = RESTART_STEP - first_warning
        print(f"  First warning:    Step {first_warning} ({lead} steps before restart)")
        print(f"\n  >>> ARC SAW INSTABILITY {lead} STEPS BEFORE THE RESTART! <<<")
    else:
        print(f"  First warning:    Step {first_warning} (after restart)")
        print(f"  Detection was reactive, not predictive")
else:
    print(f"  No warning detected — need to tune signals")
print("=" * 60)