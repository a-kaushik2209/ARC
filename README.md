# ARC - Automatic Recovery Controller

[![PyPI version](https://badge.fury.io/py/arc-training.svg)](https://badge.fury.io/py/arc-training)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Tests](https://github.com/aryankaushik/arc-training/actions/workflows/tests.yml/badge.svg)](https://github.com/aryankaushik/arc-training/actions)

**Auto-detect and recover from neural network training failures.**

ARC automatically detects NaN/Inf losses, gradient explosions, OOM errors, and silent failures—then recovers your training without losing progress.

## Key Results

| Metric             | ARC v4.0              |
| ------------------ | --------------------- |
| **Recovery Rate**  | 100% (on test suite)  |
| **Overhead**       | 27% (ARC Lite)        |
| **vs torchft**     | 3/3 vs 1/3 recoveries |
| **Max Model Size** | 1.5B params           |

## Configurations

| Config       | Overhead | Use Case                |
| ------------ | -------- | ----------------------- |
| **ARC Lite** | 27%      | Production training     |
| ARC Full     | 44%      | Debugging unstable runs |

## What ARC Handles

| Failure Type       | Detection | Recovery | Status         |
| ------------------ | --------- | -------- | -------------- |
| NaN/Inf Loss       | ✅        | ✅       | Validated      |
| Loss Explosion     | ✅        | ✅       | Validated      |
| Gradient Explosion | ✅        | ✅       | Validated      |
| OOM (all stages)   | ✅        | ✅       | Validated      |
| Accuracy Collapse  | ✅        | ⚠️       | Detection only |
| Mode Collapse      | ✅        | ⚠️       | Detection only |

## Benchmarks

```
Recovery Rate: 100% (160/160 induced failures)
Statistical Significance: p < 0.001
Models Tested: CNN, ViT, Transformer, Diffusion (up to 1.5B params)
```

## Links

- [Documentation](https://arc-training.readthedocs.io)
- [Paper](link-to-paper)
- [Efficiency Report](ARC_EFFICIENCY_REPORT.md)

## Citation

```bibtex
@software{arc2026,
  title={ARC: Automatic Recovery Controller for Neural Network Training},
  author={Kaushik, Aryan},
  year={2026},
  url={https://github.com/aryankaushik/arc-training}
}
```

## 📄 License

AGPL-3.0 License - see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Aryan Kaushik. All rights reserved.
