# Changelog

All notable changes to ARC (Automatic Recovery Controller) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2026-01-28

### Highlights

- **27% overhead** with ARC Lite configuration (validated)
- **100% recovery** on numeric failures (NaN/Inf/explosion)
- **3/3 vs 1/3** recovery rate compared to torchft
- **1.5B parameter** models validated (GPT-2 XL)

### Added

- ARC Lite mode with 27% overhead (production recommended)
- OOM recovery for all training stages
- Quantized FP16 checkpoints (50% memory reduction)
- Silent failure detection (accuracy collapse, mode collapse, dead neurons)
- Head-to-head comparison with torchft
- Statistical validation (10 seeds, p < 0.001)

### Changed

- Restructured overhead section to lead with ARC Lite
- Scoped "100% recovery" to test suite of induced failures
- Updated failure types to "6 validated + 6 detection-only"

### Fixed

- Memory leaks in checkpoint restoration
- RNG state preservation across rollbacks
- Optimizer momentum handling during recovery

## [3.0.0] - 2025-12-01

### Added

- Multi-architecture support (CNN, ViT, Transformer, Diffusion)
- Scaling to 387M parameters
- Basic OOM detection

### Changed

- Improved checkpoint efficiency
- Better loss explosion detection

## [2.0.0] - 2025-09-01

### Added

- Initial public release
- Basic NaN/Inf detection
- Weight rollback with LR reduction

## [1.0.0] - 2025-06-01

### Added

- Proof-of-concept implementation
- Single-model validation
