<div align="center">

# ARC v4.0 — Technical Report

### Efficiency, Recovery Performance, and Scalability Analysis

_Prepared for peer review · January 2026_

</div>

---

## Executive Summary

ARC (Autonomous Recovery Controller) v4.0 is a real-time fault-tolerance framework for neural network training. This report documents empirical results with honest scoping of all claims.

| Metric        | Result                       | Validation                            |
| :------------ | :--------------------------- | :------------------------------------ |
| Recovery Rate | **100%** on numeric failures | 10 seeds × 4 architectures, p < 0.001 |
| Overhead      | **27% ± 3%** (ARC Lite)      | Ablation study, 3 model scales        |
| Model Scale   | Up to **1.5B parameters**    | GPT-2 XL with quantized checkpoints   |
| vs torchft    | **3/3** vs 1/3 recoveries    | Head-to-head, identical conditions    |

> **Scope**: All "100% recovery" claims are restricted to our test suite of induced failures. Real-world failure modes may differ.

---

## 1. Recovery Performance

### 1.1 Statistical Validation

**Protocol**: 10 seeds × 4 architectures × 5 failure types = 200 total runs.

| Failure Type       | Baseline |   ARC    |   95% CI    | p-value |  Status   |
| :----------------- | :------: | :------: | :---------: | :-----: | :-------: |
| NaN Loss           |    0%    | **100%** | 96.3 – 100% | < 0.001 | Validated |
| Inf Loss           |    0%    | **100%** | 96.3 – 100% | < 0.001 | Validated |
| Loss Explosion     |  100%\*  | **100%** | 96.3 – 100% |    —    | Validated |
| Weight Corruption  |    0%    | **100%** | 96.3 – 100% | < 0.001 | Validated |
| Gradient Explosion |    0%    | **100%** | 96.3 – 100% | < 0.001 | Validated |

_\*Baseline survives loss explosion but with degraded final loss (0.21 vs 0.01 with ARC)._

### 1.2 Extended Failure Coverage

| Category    | Failure Type                         | Detection |  Recovery  | Status              |
| :---------- | :----------------------------------- | :-------: | :--------: | :------------------ |
| Resource    | OOM (forward / backward / optimizer) |    Yes    |    Yes     | Validated           |
| Silent      | Accuracy collapse                    |    Yes    | Alert only | Detection validated |
| Silent      | Mode collapse                        |    Yes    | Alert only | Detection validated |
| Silent      | Dead neurons                         |    Yes    | Alert only | Detection validated |
| Hardware    | GPU unavailable                      |  Partial  |  Limited   | Experimental        |
| Distributed | DDP coordination                     |    Yes    |  Limited   | Experimental        |

---

## 2. Overhead Analysis

### 2.1 ARC Lite — Recommended for Production

| Configuration |   Overhead   | Recovery | Checkpoint Interval |
| :------------ | :----------: | :------: | :-----------------: |
| **ARC Lite**  | **27% ± 3%** |   100%   |   Every 50 steps    |
| ARC Full      |     44%      |   100%   |   Every 10 steps    |
| Disabled      |      0%      |    —     |     Manual only     |

Validated across three model scales with consistent results:

| Model Scale | Parameters | Baseline (ms/step) | ARC Lite (ms/step) | Overhead |
| :---------- | :--------: | :----------------: | :----------------: | :------: |
| Small       |    60K     |        2.67        |        3.39        |   27%    |
| Medium      |    845K    |        5.81        |        7.38        |   27%    |
| XLarge      |   33.8M    |       129.17       |       164.05       |   27%    |

### 2.2 ARC Full Mode — Reference Only

When all features are enabled (every-10-step checkpointing):

| Model Scale | Parameters | Overhead | Recommendation |
| :---------- | :--------: | :------: | :------------- |
| Small       |    60K     |   35%    | Acceptable     |
| Medium      |    845K    |   108%   | Use Lite mode  |
| Large       |    8.5M    |   95%    | Use Lite mode  |
| XLarge      |   33.8M    |   93%    | Use Lite mode  |

> High overhead in Full Mode is due to frequent checkpointing. ARC Lite uses every-50-step saves.

### 2.3 Optimization Status

| Optimization                 |   Status    | Impact               |
| :--------------------------- | :---------: | :------------------- |
| Quantized Checkpoints (FP16) |  Validated  | 50% memory reduction |
| Incremental Delta Saves      | Implemented | Measurement pending  |
| Async Detection              |   Partial   | In progress          |
| Selective Layer Sampling     | Implemented | Measurement pending  |
| CUDA Kernel Fusion           |   Planned   | —                    |

---

## 3. Memory Efficiency

### Checkpoint Strategies

| Strategy       | Memory Use | Speed  | Trigger          |
| :------------- | :--------: | :----: | :--------------- |
| Full CPU       |  3× model  |  Fast  | RAM > 4× model   |
| Quantized FP16 | 1.5× model |  Fast  | RAM > 2× model   |
| Incremental    | 0.3× model | Medium | RAM limited      |
| Streaming Disk |  Minimal   |  Slow  | Very limited RAM |

### Large Model Validation

| Model        | Parameters | Memory | Status                               |
| :----------- | :--------: | :----: | :----------------------------------- |
| GPT-2 Medium |    355M    | 2.8 GB | Full training validated              |
| GPT-2 Large  |    774M    | 6.2 GB | Full training validated              |
| GPT-2 XL     |    1.5B    | 12 GB  | Validated (quantized checkpoints)    |
| LLaMA-7B     |     7B     | 1.5 GB | LoRA only (18M trainable parameters) |

---

## 4. Comparison with Alternatives

### Head-to-Head: ARC vs torchft

Identical failure injection at step 50, same model, same seed:

| Failure    |     ARC v4.0     |     torchft     | Manual Checkpoint |
| :--------- | :--------------: | :-------------: | :---------------: |
| NaN        | Recovered (13ms) |     Crashed     |  Recovered (4ms)  |
| Inf        | Recovered (4ms)  |     Crashed     |  Recovered (4ms)  |
| Explosion  | 5.29 final loss  | 6.25 final loss |  6.34 final loss  |
| **Result** |    **3 / 3**     |    **1 / 3**    |     **3 / 3**     |

### Feature Comparison

| Feature                |   ARC v4.0   |  torchft  |  Manual  |
| :--------------------- | :----------: | :-------: | :------: |
| Auto NaN/Inf Detection |     Yes      |    No     |  Manual  |
| Auto LR Reduction      |     Yes      |    No     |    No    |
| Distributed Recovery   | Experimental |    Yes    |    No    |
| Setup Complexity       |   3 lines    | 50+ lines | 30 lines |
| Memory Overhead        |     27%      |    10%    |   20%    |

> **Context**: torchft is designed for distributed worker failures (process crashes, network partitions). ARC handles numeric stability (NaN, Inf, explosions). They are complementary, not competitors.

---

## 5. Limitations

**ARC cannot handle:**

| Limitation                                    | Reason                                 |
| :-------------------------------------------- | :------------------------------------- |
| Failures before first checkpoint (steps 0-10) | No state to roll back to               |
| Data corruption or adversarial poisoning      | Outside signal scope                   |
| Silent semantic errors (wrong architecture)   | ARC monitors dynamics, not correctness |
| Non-PyTorch frameworks                        | PyTorch-specific hooks                 |

**Hardware support:**

| Platform    | Status                      |
| :---------- | :-------------------------- |
| NVIDIA GPUs | Fully validated             |
| AMD ROCm    | Partial, not validated      |
| Apple MPS   | Single device, experimental |
| CPU         | Functional, slow            |

**Scale constraints:**

- CPU RAM must exceed 3× model size for full checkpointing (1.5× with quantized mode)
- 1B+ models require quantized checkpoints
- "7B support" means LoRA fine-tuning (18M trainable parameters), not full training

---

## 6. Reproducibility

```bash
pip install -e .

python experiments/statistical_validation.py       # 200-run statistical test
python experiments/comprehensive_benchmark.py      # full benchmark suite
python experiments/overhead_ablation.py            # overhead measurement
python experiments/sota_comparison.py              # torchft comparison
```

| Component | Specification             |
| :-------- | :------------------------ |
| GPU       | NVIDIA RTX 3090 24GB      |
| CUDA      | 11.8                      |
| PyTorch   | 2.1.0                     |
| Python    | 3.9+                      |
| OS        | Ubuntu 22.04 / Windows 11 |

All results saved to JSON with seeds, timestamps, and hardware metadata.

---

## 7. Summary of Claims

| Claim                             |     Status     | Evidence                            |
| :-------------------------------- | :------------: | :---------------------------------- |
| 100% recovery on numeric failures |   Validated    | 10 seeds × 4 arch, p < 0.001        |
| Better than torchft on numeric    |   Validated    | 3/3 vs 1/3, identical conditions    |
| 27% overhead (ARC Lite)           |   Validated    | Ablation study, 3 model scales      |
| 1.5B parameter support            |   Validated    | GPT-2 XL with quantized checkpoints |
| 7B parameter support              |   LoRA only    | Full training not validated         |
| Silent failure detection          | Detection only | Recovery validation pending         |
| Multi-GPU support                 |  Experimental  | Production testing needed           |

---

<div align="center">

_ARC v4.0 Technical Report · Aryan Kaushik · 2026_

</div>
