<div align="center">

# Changelog

All notable changes to ARC are documented here.

Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) · [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

</div>

---

## [4.1.0] — 2026-03-08

### Highlights

- **Every paper claim backed by experiment scripts** — zero fabricated data
- **97.5% prediction accuracy** (up from 86.5%) with 12-feature MLP classifier
- **Real overhead measurement**: <10% for models above 250K parameters
- **Honest ablation table** from real experiment data

### Added

- `experiments/baseline_comparison.py` — 4 methods × 5 failures × 5 seeds (backs Table 4)
- `experiments/prediction_200_v2.py` — 200 scenarios, 12 features, MLP + LogReg, 5-fold CV (backs §4.2)
- `experiments/ablation_experiment.py` — 6 configs × 7 failures × 5 seeds (backs Table 5)
- `experiments/overhead_measurement.py` — per-component timing, 3 model scales (backs Table 9)
- Baseline comparison table in paper (ARC 100% vs loss-only 80% vs grad-clip 20%)
- Stress test protocol description with explicit verification criteria

### Changed

- Paper prediction: 86.5% accuracy / F1=0.844 → **97.5% / F1=0.974** (12 features + MLP)
- Paper ablation table: replaced fabricated numbers with real experiment data
- Paper overhead table: replaced fabricated numbers with measured data (weight stats = 77% of overhead)
- Abstract/intro/conclusion: overhead claim corrected from "below 3%" to "below 10% for 250K+ params"
- Paper methodology: trimmed ~30% (cut textbook Fisher/Lyapunov/Conformal/EWC theory)
- README.md: complete rewrite with honest, verified numbers
- All 19 paper claims rewritten with honest language

### Fixed

- Conclusion referenced fabricated "42% single-signal" claim — now matches real ablation
- Stress test table lacked protocol description — added full procedure

---

## [4.0.0] — 2026-01-28

### Highlights

- **27% overhead** with ARC Lite configuration (production-recommended)
- **100% recovery** on all numeric failure types (NaN, Inf, explosion, corruption)
- **3/3 vs 1/3** head-to-head win against torchft on numeric stability
- **1.5B parameter** model validation (GPT-2 XL with quantized checkpoints)

### Added

- ARC Lite preset — 27% overhead with full recovery, recommended for production use
- OOM recovery across all training stages (forward, backward, optimizer step)
- Quantized FP16 checkpointing — 50% memory reduction with no recovery degradation
- Silent failure detection suite (accuracy collapse, mode collapse, dead neurons)
- Head-to-head comparison framework against torchft
- Statistical validation harness (10 seeds × 4 architectures, p < 0.001)
- Mamba-based failure predictor with Evidential Deep Learning uncertainty
- Signal pipeline: 16+ real-time training health signals
- Fisher Information-based parameter importance tracking
- Multi-scale temporal fusion (5/20/50 step analysis windows)
- REST/gRPC monitoring API with Prometheus metrics

### Changed

- Restructured public API: `Arc` class replaces `WeightRollback` as primary entry point
- `ArcCallback` replaces `ARCCallback` for Lightning integration
- Overhead section rewritten to lead with validated ARC Lite numbers
- Recovery claims scoped to test suite of induced failures
- Failure type taxonomy: 6 validated + 6 detection-only

### Fixed

- Memory leaks during checkpoint restoration on repeated rollbacks
- RNG state preservation across rollback boundaries
- Optimizer momentum buffer handling during recovery
- Gradient accumulation state corruption after rollback

---

## [3.0.0] — 2025-12-01

### Added

- Multi-architecture support (CNN, ViT, Transformer, Diffusion UNet)
- Scaling validated to 387M parameters
- Basic OOM detection and graceful degradation

### Changed

- Checkpoint system restructured for incremental delta support
- Loss explosion detection threshold made adaptive

---

## [2.0.0] — 2025-09-01

### Added

- Initial public release
- NaN/Inf loss detection with automatic rollback
- Weight rollback with configurable LR reduction

---

## [1.0.0] — 2025-06-01

### Added

- Proof-of-concept implementation
- Single-model, single-failure-type validation
