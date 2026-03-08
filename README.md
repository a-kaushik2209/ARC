<div align="center">

# ARC

### Autonomous Recovery Controller for Neural Network Training

_Real-time fault tolerance that monitors, predicts, and recovers from training failures — automatically._

[![PyPI](https://img.shields.io/badge/PyPI-arc--training-blue?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/arc-training)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-green?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)

---

**3 lines of code** · **27% overhead** · **100% recovery on numeric failures** · **Up to 1.5B parameters**

[Quick Start](#quick-start) · [Architecture](#architecture) · [Benchmarks](#benchmarks) · [Documentation](#documentation)

</div>

---

## The Problem

Training neural networks is fragile. A single NaN gradient, an OOM spike, or an exploding loss at hour 47 of a 48-hour run can destroy days of compute. Engineers waste enormous time adding manual checkpointing, writing recovery scripts, and babysitting long runs.

**ARC eliminates this entirely.** It wraps your training loop with an autonomous controller that:

1. **Monitors** — Tracks 16+ real-time signals (gradient norms, loss curvature, Fisher Information, activation health, weight dynamics)
2. **Predicts** — Uses a Mamba-based state-space model with Evidential Deep Learning to estimate failure probability before it happens
3. **Recovers** — Automatically rolls back to the last healthy checkpoint and adjusts learning rate when failures are detected

You keep training. ARC keeps it alive.

---

## Quick Start

### Installation

```bash
pip install arc-training
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

That's it. ARC handles NaN detection, gradient explosion recovery, OOM fallback, checkpoint management, and learning rate adjustment — all behind `controller.step()`.

### PyTorch Lightning

```python
from arc import ArcCallback

trainer = pl.Trainer(callbacks=[ArcCallback()])
```

---

## Architecture

ARC is not a single heuristic — it's a modular system with 16 sub-packages:

```
arc/
├── core/            Self-healing engine with rollback + LR reduction
├── signals/         16+ signal collectors (gradient, activation, weight, loss, optimizer)
├── features/        Feature extraction, normalization, and buffering
├── prediction/      Failure prediction with uncertainty quantification
├── learning/        Mamba-based meta-model + trajectory simulation for predictor training
├── intervention/    12 recovery strategies (LR reduction, gradient clipping, reinitialization, ...)
├── checkpointing/   Quantized FP16 checkpoints, incremental deltas, streaming disk fallback
├── distributed/     Experimental multi-GPU coordination
├── introspection/   Fisher Information, Hessian approximation, loss landscape analysis
├── physics/         Training dynamics modeling
├── uncertainty/     Evidential Deep Learning for calibrated confidence
├── evaluation/      Comprehensive benchmarking and validation
├── security/        Integrity verification for checkpoints
└── api/             REST/gRPC monitoring, Prometheus metrics, live dashboards
```

### Signal Pipeline

```
Training Step
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Signal Collectors                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Gradient  │ │ Loss     │ │ Weight   │ │ Activation    │  │
│  │ Norm/Entropy│ Curvature│ │ Update   │ │ Health/Dead % │  │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └──────┬────────┘  │
│        └──────┬──────┴──────┬─────┘             │           │
│               ▼             ▼                   ▼           │
│         Feature Extractor + Online Normalizer               │
│               │                                             │
│               ▼                                             │
│    ┌─────────────────────┐    ┌──────────────────────────┐  │
│    │  Heuristic Detector │    │  Mamba Predictor (SSM)   │  │
│    │  (instant response) │    │  (temporal pattern model) │  │
│    └─────────┬───────────┘    └────────────┬─────────────┘  │
│              └──────────┬─────────────────┘                 │
│                         ▼                                   │
│              Risk Assessment + Recommendation               │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │     Risk < 0.3        │──── Continue training
              │     Risk 0.3 - 0.7    │──── Reduce LR, increase monitoring
              │     Risk > 0.7        │──── Rollback to checkpoint + reduce LR
              └───────────────────────┘
```

---

## Failure Coverage

| Category     | Failure Type                     | Detection               | Recovery                         | Validation                          |
| ------------ | -------------------------------- | ----------------------- | -------------------------------- | ----------------------------------- |
| **Numeric**  | NaN / Inf Loss                   | Instant                 | Rollback + LR reduction          | Statistically validated (p < 0.001) |
| **Numeric**  | Loss Explosion                   | Instant                 | Rollback + LR reduction          | Statistically validated (p < 0.001) |
| **Numeric**  | Gradient Explosion               | Instant                 | Rollback + gradient clipping     | Statistically validated (p < 0.001) |
| **Numeric**  | Weight Corruption                | Instant                 | Rollback from checkpoint         | Statistically validated (p < 0.001) |
| **Resource** | OOM (forward/backward/optimizer) | Instant                 | Batch reduction + memory cleanup | Validated                           |
| **Silent**   | Accuracy Collapse                | Trend detection         | Detection + alert                | Detection validated                 |
| **Silent**   | Mode Collapse                    | Distribution monitoring | Detection + alert                | Detection validated                 |
| **Silent**   | Dead Neurons                     | Activation monitoring   | Detection + alert                | Detection validated                 |

---

## Benchmarks

### Recovery Performance

**Test matrix**: 10 seeds × 4 architectures × 5 failure types = 200 runs

| Failure Type       | Baseline Recovery | ARC Recovery | 95% Confidence Interval | p-value |
| ------------------ | :---------------: | :----------: | :---------------------: | :-----: |
| NaN Loss           |        0%         |   **100%**   |       96.3 – 100%       | < 0.001 |
| Inf Loss           |        0%         |   **100%**   |       96.3 – 100%       | < 0.001 |
| Loss Explosion     |      100%\*       |   **100%**   |       96.3 – 100%       |    —    |
| Weight Corruption  |        0%         |   **100%**   |       96.3 – 100%       | < 0.001 |
| Gradient Explosion |        0%         |   **100%**   |       96.3 – 100%       | < 0.001 |

_\*Baseline survives explosion but with degraded final loss (0.21 vs 0.01 with ARC)_

### Head-to-Head: ARC vs torchft

Identical failure injection at step 50, same model, same seed:

| Failure   |      ARC v4.0       |     torchft     | Manual Checkpoint |
| --------- | :-----------------: | :-------------: | :---------------: |
| NaN       |  Recovered (13ms)   |     Crashed     |  Recovered (4ms)  |
| Inf       |   Recovered (4ms)   |     Crashed     |  Recovered (4ms)  |
| Explosion | **5.29** final loss | 6.25 final loss |  6.34 final loss  |
| **Score** |      **3 / 3**      |    **1 / 3**    |     **3 / 3**     |

> **Note**: torchft is designed for distributed worker failures (process crashes, network partitions), not numeric stability. ARC fills the gap torchft intentionally doesn't address. They are complementary tools.

### Overhead

| Configuration |   Overhead   | Recovery | Recommendation          |
| :------------ | :----------: | :------: | :---------------------- |
| **ARC Lite**  | **27% ± 3%** |   100%   | Production training     |
| ARC Full      |     44%      |   100%   | Debugging unstable runs |

Validated across Small (60K), Medium (845K), and XLarge (33.8M) parameter models.

### Model Scale

| Model                   | Parameters | Status                                |
| :---------------------- | :--------- | :------------------------------------ |
| YOLOv11                 | 2.6M       | Validated                             |
| DINOv2-Small            | 21M        | Validated                             |
| Llama-Style Transformer | 33.8M      | Validated                             |
| Stable Diffusion UNet   | 33.8M      | Validated                             |
| GPT-2 Medium            | 355M       | Validated                             |
| GPT-2 Large             | 774M       | Validated                             |
| GPT-2 XL                | 1.5B       | Validated (quantized checkpoints)     |
| LLaMA-7B                | 7B         | LoRA fine-tuning only (18M trainable) |

---

## Configuration

ARC works out of the box, but every parameter is tunable:

```python
from arc import Arc
from arc.config import Config

config = Config()
config.monitoring.check_interval = 50        # steps between deep checks
config.checkpointing.quantized = True        # FP16 checkpoints (50% memory)
config.recovery.max_rollbacks = 5            # max consecutive rollbacks
config.recovery.lr_reduction_factor = 0.5    # LR multiplier after recovery

controller = Arc(model, optimizer, config=config)
```

### Presets

```python
# Production: minimal overhead, maximum stability
controller = Arc(model, optimizer, preset="lite")

# Debug: frequent checkpoints, verbose logging
controller = Arc(model, optimizer, preset="full")
```

---

## Known Limitations

ARC is honest about what it cannot do:

- **First 10 steps**: No checkpoint exists yet — failures before the first save are unrecoverable
- **Data problems**: ARC cannot detect data corruption, label noise, or adversarial poisoning
- **Semantic errors**: Wrong architecture, bad hyperparameters, or flawed loss functions are outside scope
- **Non-PyTorch**: Only PyTorch is supported
- **Distributed**: Multi-GPU support is experimental and not production-validated
- **Memory**: Full checkpointing requires CPU RAM > 3× model size (use quantized mode to halve this)

---

## Technical Foundation

ARC's prediction engine is built on:

- **Mamba (Selective State Space Model)** — Captures long-range temporal dependencies in training signal sequences with O(n) complexity
- **Evidential Deep Learning** — Produces calibrated uncertainty estimates (Dirichlet priors over failure mode probabilities), distinguishing "confidently safe" from "uncertain"
- **Fisher Information Matrix** — Estimates parameter importance to prioritize which weights to monitor and which to checkpoint
- **Multi-Scale Temporal Fusion** — Analyzes training dynamics at multiple time scales (5, 20, 50 step windows) simultaneously

---

## Documentation

| Document                                      | Description                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [Efficiency Report](ARC_EFFICIENCY_REPORT.md) | Full benchmark data with statistical validation, overhead analysis, and reviewer responses |
| [Deployment Guide](DEPLOYMENT_GUIDE.md)       | PyPI publishing, Docker, CI/CD, and distribution setup                                     |
| [Changelog](CHANGELOG.md)                     | Version history and release notes                                                          |
| [Examples](examples/)                         | Integration examples for vanilla PyTorch, Lightning, and MNIST                             |

---

## Reproducibility

All benchmark results are fully reproducible:

```bash
git clone https://github.com/a-kaushik2209/ARC.git
cd ARC
pip install -e .

python experiments/statistical_validation.py      # 200-run statistical test
python experiments/comprehensive_benchmark.py     # full benchmark suite
python experiments/sota_comparison.py             # torchft comparison
```

Results are saved as JSON files with seeds, timestamps, and hardware metadata.

**Environment**: Python 3.9+ · PyTorch 2.1+ · NVIDIA GPU (validated), CPU (supported)

---

## Citation

```bibtex
@software{arc2026,
  title   = {ARC: Autonomous Recovery Controller for Neural Network Training},
  author  = {Kaushik, Aryan},
  year    = {2026},
  url     = {https://github.com/a-kaushik2209/ARC},
  version = {4.0.0}
}
```

---

<div align="center">

**AGPL-3.0 License** · Copyright (c) 2026 Aryan Kaushik

_Built to make neural network training unkillable._

</div>
