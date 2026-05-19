<div align="center">

# ARC

### Autonomous Recovery Controller for Neural Network Training

> Real-time fault tolerance for deep learning systems — detect failures early, recover automatically, and keep training alive.

[![PyPI](https://img.shields.io/badge/PyPI-arc--training-blue?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/arc-training)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-AGPL_v3-green?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)

---

### Built for resilient training pipelines

**3-line integration** • **Automatic rollback** • **Multi-signal monitoring** • **Validated on models up to 117M parameters**

[Quick Start](#quick-start) • [Architecture](#architecture) • [Benchmarks](#benchmarks) • [Contributing](#contributing)

</div>

---

# Why ARC?

Training modern neural networks is fragile.

A single NaN loss, exploding gradient, optimizer corruption, or sudden instability can destroy hours — sometimes days — of compute. Most engineers solve this manually with:

- periodic checkpoints
- custom recovery scripts
- gradient clipping hacks
- constant monitoring during long runs

ARC automates this process.

It acts as an autonomous recovery layer around your training loop that continuously:

- monitors training health
- predicts instability before collapse
- restores healthy states automatically
- applies corrective interventions when needed

The goal is simple:

> Your model trains. ARC handles survival.

---

# Features

## Real-Time Failure Detection

ARC continuously tracks:

- Gradient norms
- Loss behaviour
- Weight stability
- Optimizer state integrity
- Oscillation patterns
- Parameter drift

---

## Autonomous Recovery

When instability is detected, ARC can:

- Roll back to the last healthy checkpoint
- Reduce learning rate automatically
- Apply gradient clipping
- Restore optimizer state
- Recover from NaN / Inf corruption

No manual intervention required.

---

## Predictive Monitoring

ARC does not only react to failures.

It predicts potential instability using:

- heuristic detectors
- statistical signals
- lightweight ML models

This allows intervention before training fully collapses.

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

### Or install from source:

```bash
git clone https://github.com/a-kaushik2209/ARC.git
cd ARC
pip install -e .
```

### 3-Line Integration

For existing training pipelines, ARC can be added with only three lines of code.

```python
from arc import Arc

controller = Arc(model, optimizer)
action = controller.step(loss)
```

That single controller.step(loss) call enables:

- Training health monitoring
- Automatic checkpointing
- Failure detection
- Rollback recovery
- Learning rate correction

without changing the rest of your training loop.

Perfect for quickly integrating ARC into existing PyTorch workflows.

---

### Minimal Working Example :

```python
from arc import Arc
import torch

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters())

controller = Arc(model, optimizer)

for batch in dataloader:

    loss = model(batch)

    action = controller.step(loss)

    # Continue only if ARC did not rollback
    if not action.rolled_back:

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

In this workflow:

- ARC monitors training stability in real time
- Unhealthy states trigger automatic rollback
- Optimizer corruption and gradient explosions are handled automatically
- Training continues without manual intervention

This keeps long-running experiments significantly more resilient and fault-tolerant.

### ARC handles:

- Failure monitoring
- Checkpointing
- Rollback recovery
- Learning rate correction
- Instability detection

---

### Example Recovery Flow :

[ARC WARNING] Gradient explosion detected
[ARC ACTION] Rolling back to checkpoint #4
[ARC ACTION] Reducing LR: 0.001 → 0.0005
[ARC STATUS] Training stabilized

---

## Architecture :

ARC follows a modular monitoring pipeline designed for extensibility and low-overhead execution. That you can see below -

arc/
├── core/ Core recovery engine
├── signals/ Signal collectors
├── features/ Feature extraction pipeline
├── prediction/ Failure prediction models
├── intervention/ Recovery strategies
├── checkpointing/ Checkpoint management
├── introspection/ Stability analysis utilities
├── physics/ Dynamic system analysis
├── uncertainty/ Confidence estimation
└── evaluation/ Benchmark and validation scripts

### Component Overview :

---

| Module           | Purpose                                                       |
| ---------------- | ------------------------------------------------------------- |
| `core/`          | Main recovery controller and rollback logic                   |
| `signals/`       | Collects metrics from gradients, weights, and optimizer state |
| `features/`      | Converts raw telemetry into structured features               |
| `prediction/`    | Predicts instability using ML-based classifiers               |
| `intervention/`  | Executes recovery strategies                                  |
| `checkpointing/` | Maintains rolling recovery checkpoints                        |
| `evaluation/`    | Benchmarking and validation framework                         |

---

---

### Signal Pipeline

ARC follows a multi-stage signal processing pipeline designed to detect instability before training completely fails.

Instead of relying on a single metric like loss, ARC combines multiple training signals to build a more reliable understanding of model health in real time.

---
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

#### Signal Collection :-

At every training step, ARC collects multiple signals directly from the training process.

These signals include:

---

| Signal               | Purpose                                 |
| -------------------- | --------------------------------------- |
| Gradient Norms       | Detect exploding or vanishing gradients |
| Loss Behaviour       | Monitor instability and divergence      |
| Weight Statistics    | Detect corruption or abnormal drift     |
| Optimizer State      | Verify optimizer integrity              |
| Oscillation Patterns | Identify unstable training dynamics     |

---

Unlike traditional monitoring systems that depend only on loss values, ARC uses a multi-signal approach for better reliability and earlier detection.

#### Feature Extraction :-

Raw training signals are transformed into structured numerical features.

ARC computes:

- Rolling averages
- Variance statistics
- Trend analysis
- Stability indicators
- Temporal patterns

This creates a compact representation of training behaviour that can be analyzed efficiently in real time.

#### Failure Prediction :-

The extracted features are passed into ARC’s prediction layer.

This layer combines:

- Heuristic Detection

Fast rule-based checks for:

- NaN values
- Inf losses
- Sudden spikes
- Gradient explosions

* ML-Based Prediction

Lightweight classifiers analyze historical signal behaviour to identify instability before catastrophic failure occurs.

This hybrid system enables:

- Fast response times
- Low false positives
- Predictive recovery instead of reactive recovery

#### Risk Assessment :-

ARC classifies the current training state into one of three categories:

---

| State   | Meaning                        |
| ------- | ------------------------------ |
| HEALTHY | Training is stable             |
| WARNING | Instability signals detected   |
| FAILURE | Recovery intervention required |

---

This decision layer determines whether training should continue normally or whether intervention is necessary.

#### Recovery Engine :-

If instability reaches critical levels, ARC automatically activates its recovery system.

Possible interventions include:

- Rolling back to the last healthy checkpoint
- Reducing learning rate
- Gradient clipping
- Optimizer state restoration
- Stabilization procedures

The objective is to recover training automatically while minimizing loss of progress.

### Why Multi-Signal Monitoring Matters ?

Many training failures are silent.

Loss values alone often fail to detect:

- Optimizer corruption
- Unstable oscillations
- Parameter drift
- Hidden divergence patterns

By combining multiple independent signals, ARC provides significantly more robust fault detection and recovery compared to traditional single-metric monitoring systems.

---

## Failure Coverage :

Modern neural network training can fail in many different ways.

Some failures are obvious — like NaN losses or exploding gradients.  
Others are far more dangerous because they happen silently in the background while training appears normal.

ARC is designed to handle both.

Instead of reacting only after a complete crash, ARC continuously monitors training health and intervenes before instability becomes irreversible.

---

### Failure Detection & Recovery Matrix -

ARC is designed to detect both immediate numerical failures and slower silent instabilities that traditional monitoring systems often miss.

 --------------------------------------------------------------------------------
| Category    | Failure Type          | Detection | Recovery                     |
| ----------- | --------------------- | --------- | ---------------------------- |
| **Numeric** | NaN / Inf Loss        | Instant   | Rollback + LR reduction      |
| **Numeric** | Loss Explosion        | Instant   | Rollback + LR reduction      |
| **Numeric** | Gradient Explosion    | Instant   | Rollback + gradient clipping |
| **Numeric** | Weight Corruption     | Instant   | Rollback from checkpoint     |
| **Silent**  | Optimizer State Reset | Detected  | Rollback + state restoration |
| **Silent**  | Silent Weight Drift   | Detected  | Alert + optional rollback    |
| **Silent**  | LR Spike              | Instant   | Rollback + LR correction     |
 --------------------------------------------------------------------------------


---

### Understanding the Failure Types:

#### NaN / Inf Loss

One of the most common catastrophic failures in deep learning.
This usually happens because of:
- Unstable gradients
- Excessive learning rates
- Numerical overflow
- Invalid mathematical operations

Once NaNs appear, training is effectively destroyed.

ARC immediately:
- Detects invalid values
- Stops unstable progression
- Restores the last healthy checkpoint
- Reduces learning rate automatically

This prevents the entire experiment from collapsing.

---

#### Loss Explosion

Sometimes training appears stable for hours before the loss suddenly diverges.
This can happen because of:
- Unstable optimization
- Bad hyperparameters
- Accumulated numerical instability

ARC continuously tracks loss trends and variance patterns to identify abnormal behaviour before the model becomes unrecoverable. Instead of letting the run fail completely, ARC restores a stable state and continues training.

---

#### Gradient Explosion

Large gradients can destabilize optimization and rapidly corrupt model weights.
ARC monitors:
- Gradient magnitude
- Sudden spikes
- Abnormal gradient behaviour

If dangerous patterns are detected, ARC can:
- Clip gradients
- Rollback checkpoints
- Reduce learning rate

before permanent instability occurs.

---

#### Weight Corruption

Sometimes model parameters themselves become corrupted due to:
- Unstable updates
- Invalid optimizer states
- Numerical instability

This type of failure is especially dangerous because it may not immediately appear in the loss. ARC continuously validates weight health and restores healthy checkpoints when corruption is detected.

---

#### Optimizer State Reset

Modern optimizers like Adam maintain internal momentum statistics. If these states are corrupted or reset unexpectedly:
- Convergence quality drops
- Training becomes unstable
- Recovery becomes difficult

ARC monitors optimizer consistency and restores optimizer states alongside model checkpoints. This ensures recovery is complete — not partial.

---

#### Silent Weight Drift

Not all failures happen instantly. Some models slowly drift toward unstable regions over thousands of steps while appearing normal on the surface.
These failures are difficult to detect using loss values alone.

ARC analyzes:
- Parameter movement
- Oscillation behaviour
- Stability trends

To identify hidden instability before catastrophic divergence occurs.

---

#### Learning Rate (LR) Spike

A sudden increase in learning rate can destabilize an otherwise healthy training run within seconds.ARC monitors learning rate behaviour in real time and immediately reacts to abnormal spikes using:
- Rollback recovery
- Automatic LR correction
- Stabilization procedures

---

### Why This Matters ?

Traditional training systems are mostly reactive. They notice failures only after:
- The model collapses
- The checkpoint is corrupted
- Or the experiment is already lost

ARC is designed to be proactive.
By combining:
- Real-time telemetry
- Predictive monitoring
- Automated recovery
- Multi-signal analysis

ARC transforms fragile training pipelines into significantly more resilient systems capable of surviving real-world instability without manual intervention.


---  


## Benchmarks :

> **All experiments were performed using reproducible scripts with fixed random seeds.**


### Baseline Comparison (25 scenarios) :

To evaluate ARC under realistic failure scenarios, multiple protection strategies were benchmarked against intentionally induced training failures.
The objective of this experiment was to measure how effectively each method could:
- Detect instability
- Recover from failure
- Avoid unnecessary false alarms

All experiments were performed using reproducible scripts with fixed random seeds to ensure consistent and reliable evaluation.



### Experimental Setup

The benchmark included:

- **4 protection methods**
- **5 failure types**
- **5 random seeds**
- **25 total experimental scenarios**

Script used:

```python
experiments/baseline_comparison.py
```


### Benchmark Results :


<div align="center"> 

| Method                | Detection Rate | Recovery Rate | False Positives | Overall Behaviour                                     |
| --------------------- | -------------- | ------------- | --------------- | ----------------------------------------------------- |
| **No Protection**     | 52.0%          | 0.0%          | 0               | Detects some visible failures but cannot recover      |
| **Gradient Clipping** | 20.0%          | 0.0%          | 0               | Limited protection against gradient instability only  |
| **Loss-Only Monitor** | 80.0%          | 80.0%         | 0               | Better detection, but misses silent failures          |
| **Full ARC**          | **100%**       | **100%**      | **0**           | Fully detected and recovered from all tested failures |

</div>


---


### Failure Prediction (200 scenarios) :


ARC does not rely only on rule-based failure detection. In addition to heuristic monitoring, ARC includes lightweight machine learning models capable of predicting instability before catastrophic failure occurs.
The purpose of this benchmark was to evaluate how accurately ARC could classify training behaviour as either:
- Stable
- Unstable / failure-prone

Using multi-signal telemetry collected during training.


#### Experimental Setup

The prediction benchmark was evaluated across:

- **4 neural network architectures**
- **5 failure types**
- **5 random seeds**
- **2 classification labels**
- **5-fold cross validation**

This resulted in a total of:

> **200 experimental scenarios**

Script used:

```python
experiments/prediction_200_v2.py
```


#### Evaluated Models

<div align="center">

| Model                                 | Description                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------- |
| **Logistic Regression (12 Features)** | Lightweight linear classifier using engineered telemetry features                |
| **MLP Predictor (12 Features)**       | Multi-layer neural predictor capable of learning non-linear instability patterns |

</div>


#### Prediction Benchmark Results

<div align="center">

| Classifier              | Accuracy         | Precision | Recall    | F1 Score         | Interpretation                                               |
| ----------------------- | ---------------- | --------- | --------- | ---------------- | ------------------------------------------------------------ |
| **Logistic Regression** | 95.5% ± 1.9%     | 100%      | 91.0%     | 0.953 ± 2.6%     | Strong baseline with highly reliable predictions             |
| **MLP Predictor**       | **97.5% ± 2.2%** | **100%**  | **95.0%** | **0.974 ± 2.8%** | Best overall performance with improved instability detection |

</div>


---


### Ablation Study (35 scenarios) :


To understand which components contribute most to ARC’s stability system, an ablation study was performed.
In this experiment, individual monitoring components were removed one at a time to measure how detection performance changed. The goal was to answer an important question:

> Which parts of ARC are truly responsible for robust failure detection?


#### Experimental Setup

The ablation benchmark included:
- **7 failure types**
- **5 random seeds**
- Multiple reduced configurations of ARC

Script used:

```python
experiments/ablation_experiment.py
```


#### Ablation Results 

<div align="center">

| Configuration                 | Detection Rate | Change from Full ARC | Interpretation                                              |
| ----------------------------- | -------------- | -------------------- | ----------------------------------------------------------- |
| **Full ARC (All Components)** | **85.7%**      | ---                  | Maximum protection with full multi-signal monitoring        |
| **− Weight Health**           | 85.7%          | 0.0%                 | Other signals compensate effectively                        |
| **− Gradient Monitoring**     | 85.7%          | 0.0%                 | Redundant protection still maintains detection              |
| **− Loss Monitoring**         | 85.7%          | 0.0%                 | Stability remains strong through alternative signals        |
| **− Optimizer State**         | 71.4%          | −14.3%               | Significant detection loss for silent failures              |
| **Loss Only (Baseline)**      | 71.4%          | −14.3%               | Traditional monitoring misses critical instability patterns |

</div>


---


### Performance Overhead :


A recovery system is only useful if it remains lightweight during training. One of ARC’s core design goals is to provide strong fault tolerance without introducing significant computational overhead.
To evaluate this, ARC’s runtime cost was measured across different monitoring components and model scales.
The benchmark focused on answering:

> **How much additional latency does ARC introduce during training?**


#### Experimental Setup

Performance measurements were collected using controlled benchmark experiments on CPU environments. The evaluation measured:

- Per-step monitoring latency
- Checkpoint overhead
- Feature extraction cost
- Prediction latency
- Scaling behaviour across model sizes

Script used:

```python
experiments/overhead_measurement.py
```


#### Component-Level Overhead

The table below shows how much time each subsystem contributes during a training step.

<div align="center">

| Component                     | Time (ms)   | Contribution to ARC Runtime | Description                                      |
| ----------------------------- | ----------- | --------------------------- | ------------------------------------------------ |
| **Gradient Norm Monitoring**  | 0.12 ms     | 9.0%                        | Tracks gradient stability and detects explosions |
| **Weight Statistics**         | 1.06 ms     | 76.9%                       | Computes parameter health and integrity metrics  |
| **Loss Analysis**             | 0.01 ms     | 0.6%                        | Monitors loss behaviour and divergence trends    |
| **Checkpointing (Amortized)** | 0.13 ms     | 9.6%                        | Maintains rolling recovery checkpoints           |
| **Forecasting / Prediction**  | 0.06 ms     | 4.1%                        | Runs heuristic and ML-based failure prediction   |
| **Total ARC Runtime**         | **1.38 ms** | **100%**                    | Full monitoring and recovery pipeline            |

</div>


#### Scaling Behaviour Across Model Sizes

ARC was also tested on models of different parameter scales to understand how monitoring overhead changes as models grow larger.

<div align="center">

| Model Scale    | Parameters | ARC Overhead | Relative Training Cost | Interpretation                               |
| -------------- | ---------- | ------------ | ---------------------- | -------------------------------------------- |
| **Small MLP**  | 50K        | 0.86 ms      | ~60%                   | Monitoring dominates very small models       |
| **Medium CNN** | 288K       | 1.38 ms      | ~10%                   | Overhead becomes significantly smaller       |
| **Large CNN**  | 2.5M       | 7.04 ms      | ~9.5%                  | ARC scales efficiently with larger workloads |

</div>


---


### Large-Scale Validation

To evaluate ARC under demanding real-world conditions, the system was tested on large-scale deep learning architectures across multiple induced failure scenarios.
The purpose of this benchmark was to verify whether ARC could remain stable and effective beyond small experimental models.

The evaluation focused on three key questions:
- Can ARC recover large models reliably?
- Does the recovery system remain stable under severe failures?
- How does ARC behave across different architectures and parameter scales?


#### Experimental Setup

The stress test included:
- Transformer architectures
- Convolutional networks
- Diffusion model components
- Vision transformers

With parameter counts ranging from:

```text
100K → 117M parameters

Script used:

```python
experiments/validate_claims_phase2.py
```


#### Large-Scale Stress Test Results

<div align="center">

| Model            | Parameters | Injected Failure          | ARC Recovery Status | Rollbacks Triggered | Interpretation                                 |
| ---------------- | ---------- | ------------------------- | ------------------- | ------------------- | ---------------------------------------------- |
| **NanoGPT**      | 10M        | Learning Rate Spike (50×) | ✓ Successful        | 2                   | Stabilized severe LR instability               |
| **ResNet-50**    | 25.6M      | Loss Singularity          | ✓ Successful        | 1                   | Recovered from catastrophic divergence         |
| **GPT-2 Small**  | 50M        | NaN Bomb                  | ✓ Successful        | 4                   | Contained repeated numerical corruption        |
| **SD-UNet**      | 60M        | Gradient Attack           | ✓ Successful        | 4                   | Recovered from aggressive gradient instability |
| **ViT-Base**     | 86M        | Inf Nuke                  | ✓ Successful        | 1                   | Detected and restored invalid parameter states |
| **GPT-2 Medium** | 117M       | NaN Bomb                  | ✓ Successful        | 3                   | Maintained stability at largest tested scale   |

</div>


---


## Theoretical Foundation :


ARC is not built around a single heuristic or handcrafted rule.
Instead, it combines multiple mathematical and stability-analysis frameworks to better understand training dynamics in real time.
The objective is not only to detect visible failures, but also to identify deeper instability patterns before catastrophic collapse occurs.

These theoretical components allow ARC to move beyond traditional loss-only monitoring systems and provide more intelligent, predictive fault tolerance.


### Core Mathematical Frameworks

<div align="center">

| Framework | Purpose in ARC | Why It Matters |
|---|---|---|
| **Fisher Information** | Estimates parameter importance during recovery | Helps preserve critical learned knowledge |
| **Lyapunov Stability Analysis** | Measures dynamic stability of parameter updates | Detects unstable optimization behaviour early |
| **FFT Oscillation Detection** | Identifies periodic instability patterns | Detects hidden oscillatory training behaviour |
| **Conformal Prediction** | Provides calibrated confidence estimates | Improves reliability of stability predictions |
| **Elastic Weight Consolidation (EWC)** | Reduces catastrophic forgetting during rollback | Preserves useful learned representations |
| **Loss Landscape Analysis** | Evaluates sharpness and instability regions | Predicts collapse-prone optimization states |

</div>


### Fisher Information

ARC uses Fisher Information to estimate how important different parameters are to the model.Some parameters contribute far more heavily to learned behaviour than others.

During recovery, ARC can use this information to:
- Prioritize stable parameter restoration
- Reduce unnecessary disruption
- Preserve critical learned knowledge

This improves recovery quality after instability events.


### Lyapunov Stability Analysis

Training dynamics can be treated as a dynamic system.
ARC applies Lyapunov-inspired stability analysis to measure whether optimization behaviour is:
- Converging
- Stable
- Diverging

This allows ARC to identify instability trends before they fully appear in loss values. Instead of reacting after failure, ARC can anticipate dangerous optimization behaviour early.


### FFT Oscillation Detection

Some training failures occur through unstable oscillations rather than immediate divergence. These oscillations may remain hidden in raw loss curves but still destabilize optimization over time.

ARC applies:
- Fast Fourier Transform (FFT)
- Frequency-domain Analysis

To detect repeating instability patterns in training dynamics.
This helps identify hidden oscillatory behaviour that traditional monitoring systems often miss.


### Conformal Prediction

Prediction systems should not only make decisions — they should also estimate confidence.

ARC uses conformal prediction techniques to provide:
- Calibrated uncertainty estimates
- Distribution-aware confidence intervals
- More reliable instability assessment

This reduces unreliable predictions and improves trust in automated interventions.


### Elastic Weight Consolidation (EWC)

Rollback recovery can sometimes disturb previously learned knowledge.

ARC incorporates concepts inspired by Elastic Weight Consolidation to reduce:
- Catastrophic forgetting
- Unnecessary parameter disruption
- Recovery-induced degradation

This helps maintain training quality even after multiple recovery cycles.


### Loss Landscape Analysis

The geometry of the loss landscape contains important stability information.

Sharp regions often indicate:
- Unstable optimization
- Sensitivity to parameter updates
- Higher collapse probability

ARC analyzes landscape sharpness to estimate whether the current optimization trajectory is entering unstable regions.
This provides another predictive signal before visible divergence occurs.


## Why Multiple Frameworks Matter ?

Training instability is complex. No single metric can fully explain:

- Optimizer corruption
- Silent divergence
- Unstable oscillations
- Catastrophic collapse

ARC combines:
- Statistical analysis
- Dynamic system theory
- Optimization research
- Uncertainty estimation
- Signal processing

To build a significantly more robust understanding of training health.
This multi-framework approach is one of the key reasons ARC can detect failures earlier and recover more reliably than traditional monitoring systems.


### Key Insight

The theoretical foundation behind ARC demonstrates an important principle:

> **Reliable fault tolerance requires understanding training dynamics from multiple perspectives simultaneously.**

By integrating ideas from:
- Optimization theory
- Signal processing
- Uncertainty estimation
- Dynamic systems
- Continual learning

ARC transforms model training from a fragile process into a significantly more resilient and self-correcting system.


---


## Common Installation Issues :

### PyTorch installation fails

Ensure your Python version is:
Python 3.8+


### CUDA not detected 

Install the correct CUDA-enabled PyTorch build:

https://pytorch.org/get-started/locally/


### ModuleNotFoundError

Try:
pip install -e .


---


## Known Limitations :

ARC is designed to improve training resilience, but like any evolving research system, it still has practical limitations.
Being transparent about these limitations is important because fault tolerance systems must be evaluated realistically — not just under ideal conditions.
The current version of ARC has been experimentally validated, but several areas still require further large-scale testing and development.


### Current Limitations

<div align="center">

| Limitation | Current Status | Impact |
|---|---|---|
| **CPU-Only Validation** | Benchmarks were performed primarily on CPU environments | GPU overhead characteristics are not yet fully validated |
| **Scale Ceiling** | Tested up to 117M parameters | Behaviour beyond this scale remains experimentally unverified |
| **Synthetic Failure Testing** | Failures were intentionally injected during experiments | Real-world organic failures may behave differently |
| **Early-Step Vulnerability** | No checkpoint exists during the first few training steps | Failures before initial checkpoint creation cannot be recovered |
| **Data-Level Failures** | Dataset corruption and label noise are not monitored | ARC focuses on training dynamics, not dataset integrity |
| **Framework Support** | PyTorch-only support currently available | TensorFlow and JAX integrations are not yet implemented |

</div>


---


## Roadmap :


**Planned improvements:**
- GPU benchmarking support
- Distributed training support
- TensorFlow backend
- Better visualization dashboard
- Real-time training telemetry UI
- Enhanced failure prediction models


---


## Citation :


If you use ARC in academic research, experiments, publications, or derivative work, please cite the project using the following BibTeX entry.
Citations help support continued research and development around resilient and fault-tolerant deep learning systems.

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

_ARC — Making neural network training resilient by default._

</div>
