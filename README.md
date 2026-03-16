<div align="center">

# ARC

### Autonomous Recovery Controller for Neural Network Training

_Real-time fault tolerance that monitors, predicts, and recovers from training failures — automatically._

[![PyPI](https://img.shields.io/badge/PyPI-arc--training-blue?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/arc-training)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-green?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)

---

**3 lines of code** · **<10% overhead (250K+ params)** · **100% recovery on induced failures** · **100K–117M parameters validated**

[Quick Start](#quick-start) · [Architecture](#architecture) · [Benchmarks](#benchmarks)

</div>

---

## The Problem

Training neural networks is fragile. A single NaN gradient, an OOM spike, or an exploding loss at hour 47 of a 48-hour run can destroy days of compute. Engineers waste enormous time adding manual checkpointing, writing recovery scripts, and babysitting long runs.

**ARC eliminates this entirely.** It wraps your training loop with an autonomous controller that:

1. **Monitors** — Tracks multi-signal telemetry (loss trajectory, gradient norms, weight health, optimizer state integrity)
2. **Predicts** — Uses signal-based classifiers (97.5% accuracy, 100% precision, zero false positives) to detect failures before they become irreversible
3. **Recovers** — Automatically rolls back to the last healthy checkpoint and applies corrective measures (LR reduction, weight perturbation)

You keep training. ARC keeps it alive.

---

## Quick Start

### Installation

```bash
pip install arc-training
```

Or install from source:

```bash
git clone https://github.com/a-kaushik2209/ARC.git
cd ARC
pip install -e .
```

### 3-Line Integration

```python
from arc import Arc

controller = Arc(model, optimizer)

for batch in dataloader:
    loss = model(batch)
    action = controller.step(loss)       # monitor + protect

    if not action.rolled_back:           # normal path
        loss.backward()
        optimizer.step()
```

That's it. ARC handles NaN detection, gradient explosion recovery, checkpoint management, and learning rate adjustment — all behind `controller.step()`.

---

## Architecture

ARC is a modular multi-signal monitoring system:

```
arc/
├── core/            Self-healing engine with rollback + LR reduction
├── signals/         Multi-signal collectors (gradient, loss, weight, optimizer state)
├── features/        Feature extraction, normalization, and buffering
├── prediction/      Signal-based failure prediction (logistic regression + MLP)
├── intervention/    Recovery strategies (LR reduction, gradient clipping, weight perturbation)
├── checkpointing/   Checkpoint management with circular buffer
├── introspection/   Fisher Information, Hessian approximation, loss landscape analysis
├── physics/         Lyapunov stability analysis, FFT oscillation detection
├── uncertainty/     Conformal prediction for calibrated stability assessment
└── evaluation/      Benchmarking and validation harness
```

### Signal Pipeline

```
Training Step
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Signal Collectors                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Gradient  │ │ Loss     │ │ Weight   │ │ Optimizer     │  │
│  │ Norm/Ent. │ │ Trend/Var│ │ Norm/NaN │ │ State Norm    │  │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └──────┬────────┘  │
│        └──────┬──────┴──────┬─────┘             │           │
│               ▼             ▼                   ▼           │
│         Feature Extractor (12 features)                     │
│               │                                             │
│               ▼                                             │
│    ┌─────────────────────┐    ┌──────────────────────────┐  │
│    │  Heuristic Detector │    │  MLP Predictor           │  │
│    │  (instant response) │    │  (97.5% acc, 0 FP)       │  │
│    └─────────┬───────────┘    └────────────┬─────────────┘  │
│              └──────────┬─────────────────┘                 │
│                         ▼                                   │
│              Risk Assessment + Recovery Decision            │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │   HEALTHY             │──── Continue training
              │   WARNING             │──── Increase monitoring, prepare checkpoint
              │   FAILURE             │──── Rollback to checkpoint + corrective action
              └───────────────────────┘
```

---

## Failure Coverage

| Category    | Failure Type          | Detection | Recovery                     |
| ----------- | --------------------- | --------- | ---------------------------- |
| **Numeric** | NaN / Inf Loss        | Instant   | Rollback + LR reduction      |
| **Numeric** | Loss Explosion        | Instant   | Rollback + LR reduction      |
| **Numeric** | Gradient Explosion    | Instant   | Rollback + gradient clipping |
| **Numeric** | Weight Corruption     | Instant   | Rollback from checkpoint     |
| **Silent**  | Optimizer State Reset | Detected  | Rollback + state restoration |
| **Silent**  | Silent Weight Drift   | Detected  | Alert + optional rollback    |
| **Silent**  | LR Spike              | Instant   | Rollback + LR correction     |

---

## Benchmarks

> **All numbers below are from reproducible experiment scripts with fixed seeds.**

### Baseline Comparison (25 scenarios)

4 methods × 5 failure types × 5 seeds. Script: `experiments/baseline_comparison.py`

| Method            | Detection | Recovery | False Positives |
| :---------------- | :-------: | :------: | :-------------: |
| No Protection     |   52.0%   |   0.0%   |        0        |
| Gradient Clipping |   20.0%   |   0.0%   |        0        |
| Loss-Only Monitor |   80.0%   |  80.0%   |        0        |
| **Full ARC**      | **100%**  | **100%** |      **0**      |

### Failure Prediction (200 scenarios)

4 architectures × 5 failure types × 5 seeds × 2 labels, 5-fold CV. Script: `experiments/prediction_200_v2.py`

| Classifier         |     Accuracy     | Precision |  Recall   |        F1        |
| :----------------- | :--------------: | :-------: | :-------: | :--------------: |
| Logistic Reg (12f) |   95.5% ± 1.9%   |   100%    |   91.0%   |   0.953 ± 2.6%   |
| **MLP (12f)**      | **97.5% ± 2.2%** | **100%**  | **95.0%** | **0.974 ± 2.8%** |

### Ablation Study (35 scenarios)

7 failure types × 5 seeds. Script: `experiments/ablation_experiment.py`

| Configuration             | Detection | Δ from Full |
| :------------------------ | :-------: | :---------: |
| Full ARC (all components) |   85.7%   |     ---     |
| − Weight Health           |   85.7%   |    0.0%     |
| − Gradient Monitoring     |   85.7%   |    0.0%     |
| − Loss Monitoring         |   85.7%   |    0.0%     |
| − Optimizer State         |   71.4%   |   −14.3%    |
| Loss Only (baseline)      |   71.4%   |   −14.3%    |

> **Defense in depth**: Weight/gradient/loss provide redundant coverage (any one catches most failures). Optimizer state monitoring is uniquely valuable for silent failures.

### Overhead (measured, CPU)

Script: `experiments/overhead_measurement.py`

| Component           | Time (ms) | % of ARC Total |
| :------------------ | :-------: | :------------: |
| Gradient Norm       |   0.12    |      9.0%      |
| Weight Statistics   |   1.06    |     76.9%      |
| Loss Analysis       |   0.01    |      0.6%      |
| Checkpoint (amort.) |   0.13    |      9.6%      |
| Forecasting         |   0.06    |      4.1%      |
| **Total ARC**       | **1.38**  |    **100%**    |

| Model Scale | Parameters | ARC Overhead | Relative |
| :---------- | :--------: | :----------: | :------: |
| Small MLP   |    50K     |   0.86 ms    |   ~60%   |
| Medium CNN  |    288K    |   1.38 ms    |   ~10%   |
| Large CNN   |    2.5M    |   7.04 ms    |  ~9.5%   |

### Large Model Stress Test

Script: `experiments/validate_claims_phase2.py`

| Model        | Params | Failure Type     | ARC Recovery | Rollbacks |
| :----------- | :----- | :--------------- | :----------: | :-------: |
| NanoGPT      | 10M    | LR Spike (50×)   |      ✓       |     2     |
| ResNet-50    | 25.6M  | Loss Singularity |      ✓       |     1     |
| GPT-2 Small  | 50M    | NaN Bomb         |      ✓       |     4     |
| SD-UNet      | 60M    | Gradient Attack  |      ✓       |     4     |
| ViT-Base     | 86M    | Inf Nuke         |      ✓       |     1     |
| GPT-2 Medium | 117M   | NaN Bomb         |      ✓       |     3     |

---

## Theoretical Foundation

ARC integrates six mathematical frameworks, each experimentally validated:

| Framework                        | Purpose                                             | Validation                                               |
| :------------------------------- | :-------------------------------------------------- | :------------------------------------------------------- |
| **Fisher Information**           | Parameter importance weighting for recovery         | 11.5× separation ratio (important vs unimportant params) |
| **Lyapunov Stability**           | Online stability estimation from parameter velocity | 10× higher exponent under instability                    |
| **FFT Oscillation Detection**    | Periodic behaviour detection in training dynamics   | 6.9× power ratio at oscillation frequency                |
| **Conformal Prediction**         | Distribution-free coverage guarantees for stability | ≥99% empirical coverage at all target levels             |
| **Elastic Weight Consolidation** | Knowledge preservation during recovery              | 0.4% lower post-recovery loss                            |
| **Loss Landscape Analysis**      | Sharpness-based instability prediction              | 12.2× higher sharpness before failure                    |

---

## Known Limitations

ARC is honest about what it cannot do:

- **CPU only (validated)**: All experiments ran on CPU. GPU overhead expected to be lower but not yet measured
- **Scale ceiling**: Validated up to 117M parameters. Behaviour above this is not empirically confirmed
- **Synthetic failures only**: All test failures were programmatically injected. Organically occurring failures are untested
- **First 10 steps**: No checkpoint exists yet — failures before the first save are unrecoverable
- **Data problems**: ARC cannot detect data corruption, label noise, or adversarial poisoning
- **Non-PyTorch**: Only PyTorch is supported

---

## Citation

```bibtex
@article{kaushik2026arc,
  title   = {ARC: Autonomous Recovery Controller for Fault-Tolerant Neural Network Training},
  author  = {Kaushik, Aryan},
  year    = {2026},
  note    = {Maharaja Agrasen Institute of Technology, New Delhi}
}
```

---

<div align="center">

**AGPL-3.0 License** · Copyright (c) 2026 Aryan Kaushik

_Built to make neural network training unkillable._

</div>
