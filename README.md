<div align="center">

# ARC

### Autonomous Recovery Controller for Neural Network Training

_Real-time fault tolerance that monitors, predicts, and recovers from training failures - automatically._

<br>

[![PyPI](https://img.shields.io/badge/PyPI-arc--training-blue?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/arc-training)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-green?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/E6UvPWC8DW)

<br>

**3 lines of code** · **<10% overhead (250K+ params)** · **100% recovery on induced failures** · **100K–117M parameters validated**

<br>

[Quick Start](#quick-start) •
[Architecture](#architecture) •
[Benchmarks](#benchmarks) •
[Community](#community) •
[Collaboration Guidelines](#collaboration-guidelines)

</div>

---

# The Problem

Training neural networks is fragile. A single NaN gradient, an OOM spike, or an exploding loss at hour 47 of a 48-hour run can destroy days of compute. Engineers waste enormous time adding manual checkpointing, writing recovery scripts, and babysitting long runs.

**ARC eliminates this entirely.** It wraps your training loop with an autonomous controller that:

1. **Monitors** - Tracks multi-signal telemetry (loss trajectory, gradient norms, weight health, optimizer state integrity)
2. **Predicts** - Uses signal-based classifiers to detect failures before they become irreversible
3. **Recovers** - Automatically rolls back to the last healthy checkpoint and applies corrective measures

You keep training. ARC keeps it alive.

---

# Quick Start

## Installation

```bash
pip install arc-training
```

Or install from source:

```bash
git clone https://github.com/a-kaushik2209/ARC.git
cd ARC
pip install -e .
```

---

## 3-Line Integration

```python
from arc import ArcV2

controller = ArcV2(model, optimizer)

for batch in dataloader:
    loss = model(batch)
    action = controller.step(loss)

    if not action.get('rolled_back', False) and action.get('recommendation') != 'rollback':
        loss.backward()
        optimizer.step()
```

ARC automatically handles:

- NaN detection
- Gradient explosion recovery
- Checkpoint management
- Learning rate correction
- Automatic rollback recovery

---

# Community

<div align="center">

## Join the ARC Community

[![Discord](https://img.shields.io/badge/Join%20the%20ARC%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/E6UvPWC8DW)

</div>

The Discord server is the primary hub for:

- Contributor discussions
- Collaboration
- Development updates
- Research discussions
- Feature proposals
- Bug reporting and debugging support

We welcome contributors, researchers, students, and developers of all experience levels.

---

# Architecture

ARC is a modular multi-signal monitoring system:

```text
arc/
├── core/            Self-healing engine with rollback + LR reduction
├── signals/         Multi-signal collectors
├── features/        Feature extraction and buffering
├── prediction/      Failure prediction models
├── intervention/    Recovery strategies
├── checkpointing/   Checkpoint management
├── introspection/   Hessian + Fisher analysis
├── physics/         Stability analysis
├── uncertainty/     Conformal prediction
└── evaluation/      Benchmarking harness
```

---

# Benchmarks

## Baseline Comparison

| Method            | Detection | Recovery | False Positives |
| :---------------- | :-------: | :------: | :-------------: |
| No Protection     |   52.0%   |   0.0%   |        0        |
| Gradient Clipping |   20.0%   |   0.0%   |        0        |
| Loss-Only Monitor |   80.0%   |  80.0%   |        0        |
| **Full ARC**      | **100%**  | **100%** |      **0**      |

---

## Failure Prediction

| Classifier          | Accuracy | Precision | Recall | F1 Score |
| :------------------ | :------: | :-------: | :----: | :------: |
| Logistic Regression |  95.5%   |   100%    | 91.0%  |   0.953  |
| **MLP Predictor**   | **97.5%**| **100%**  | **95%**| **0.974**|

---

## Overhead

| Model Scale | Parameters | ARC Overhead | Relative |
| :---------- | :--------: | :----------: | :------: |
| Small MLP   |    50K     |   0.86 ms    |   ~60%   |
| Medium CNN  |    288K    |   1.38 ms    |   ~10%   |
| Large CNN   |    2.5M    |   7.04 ms    |  ~9.5%   |

---

# Collaboration Guidelines

We welcome open-source contributions, architecture improvements, benchmarking extensions, research discussions and documentation enhancements.

---

## Contribution Workflow

### 1. Fork the Repository

```bash
git clone https://github.com/a-kaushik2209/ARC.git
cd ARC
```

---

### 2. Open or Discuss an Issue First

Before starting implementation work, contributors are expected to:

- Open a new issue describing the proposed feature, improvement or bug fix
- Participate in issue discussions if clarification is required
- Wait for the issue to be assigned before beginning work

Pull requests raised without prior discussion or assignment may be closed if the contribution does not align with the current roadmap, architecture direction or active development priorities.

This process helps avoid duplicated work and keeps contributions coordinated across the project.

---

### 3. Create a Development Branch

```bash
git checkout -b feature/your-feature-name
```

Use clear and descriptive branch names.

Examples:

```text
feature/failure-prediction-upgrade
fix/checkpoint-memory-leak
docs/installation-guide-update
```

---

### 4. Install Dependencies

```bash
pip install -e .
```

Install all required dependencies before starting development to ensure local testing consistency.

---

### 5. Make Your Changes

Please keep contributions:

- Modular
- Well documented
- Consistent with the existing architecture
- Focused on a single feature or fix

For larger changes, keep commits logically separated wherever possible.

---

### 6. Commit Clearly

```bash
git commit -m "Add: short feature description"
```

Prefer concise and meaningful commit messages.

Examples:

```text
Add: gradient anomaly recovery module
Fix: checkpoint rollback edge case
Docs: improve installation instructions
```

---

### 7. Push Your Branch

```bash
git push origin feature/your-feature-name
```

Ensure your branch is up to date with the latest repository changes before opening a pull request.

---

### 8. Open a Pull Request

Please include:

- A clear explanation of the contribution
- Motivation behind the change
- Relevant benchmarks, logs or screenshots if applicable
- References to the related issue

PRs should remain focused and avoid unrelated modifications.

---

## Pull Request Standards

To maintain repository quality:

- Keep PRs focused and minimal
- Avoid unrelated refactors
- Ensure code executes correctly before submission
- Add comments and docstrings where appropriate
- Maintain compatibility with the existing architecture
- Discuss major architectural changes before implementation

Submissions that significantly change project direction without prior discussion may be deferred or closed.

---

## Issue Reporting

When opening an issue, include:

- Clear problem description
- Expected behaviour
- Steps to reproduce
- Relevant logs or screenshots
- Environment details if applicable

Bug reports with reproducible examples are highly appreciated.

---

## Development Principles

ARC prioritizes:

- Reliability over complexity
- Measured experimental validation
- Transparent benchmarking
- Modular system design
- Honest reporting of limitations

Contributions aligned with these principles are strongly encouraged.

---

## Contributor Communication

<div align="center">

[![Discord](https://img.shields.io/badge/ARC%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/E6UvPWC8DW)

</div>

The Discord server is the primary hub for:

- Contributor discussions
- Collaboration
- Development updates
- Research discussions
- Feature proposals
- Bug reporting and debugging support

Contributors are encouraged to participate in discussions before beginning larger implementations or architectural changes.

---

# Known Limitations

- CPU-only validation currently
- Validated up to 117M parameters
- Synthetic failures only
- First checkpoint gap
- No data corruption detection
- PyTorch-only support

---

# Citation

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

### AGPL-3.0 License

Copyright (c) 2026 Aryan Kaushik

_Built to make neural network training unkillable._

</div>
