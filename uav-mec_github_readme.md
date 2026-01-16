# UAV-MEC Simulation: A High-Fidelity Scenario for Value-Heterogeneous Coordination

## 📖 Overview

This repository hosts the **Custom UAV-MEC (Mobile Edge Computing) Simulation Environment**, developed as a specialized experimental testbed for multi-agent reinforcement learning (MARL) research in the **Low-Altitude Economy**.

This scenario is explicitly designed to verify algorithmic performance under **Dual System Constraints**, going beyond standard simplified models:

1. **Epistemic Fragmentation**: Partial observability induced by dynamic channel occlusion and sensor noise.
2. **Value Heterogeneity**: A realistic mix of high-stakes (Latency-Sensitive) and best-effort (Latency-Tolerant) tasks.

This simulation serves as the **System-Level Verification Scenario** for our research on *Generative Consensus* and *Value-Aware Coordination*.

------

## 🏗️ Scenario Design & Dynamics

The environment simulates a dynamic cluster of UAVs providing edge computing services to ground users (GUs) in complex terrains (e.g., Urban, Emergency zones). The physics and communication models follow the protocols established in our previous work.

### 1. Heterogeneous Task Model (The "Value Trap")

To evaluate whether agents can identify and prioritize critical goals amidst low-value distractions, tasks are categorized into two distinct types:

| **Task Type**              | **Description**                                              | **Value Weight**                | **QoS Constraint**               |
| -------------------------- | ------------------------------------------------------------ | ------------------------------- | -------------------------------- |
| **LS (Latency-Sensitive)** | Critical computations (e.g., control signals, emergency data). | **High** ($\times 10 \sim 100$) | Hard Deadline ($T_{max} < 50ms$) |
| **LT (Latency-Tolerant)**  | Background traffic (e.g., routine logs, monitoring).         | **Low** ($\times 1$)            | Best-effort                      |

- **Challenge**: Agents must learn to prioritize LS tasks to minimize the **LS Violation Rate**, avoiding the "gradient dilution" caused by the abundant LT tasks.

### 2. Fragmented Perception Model

Unlike ideal communication models, this scenario introduces realistic imperfections:

- **Dynamic Occlusion**: Communication links (LoS/NLoS) are probabilistically blocked based on terrain features, creating severe partial observability.
- **Sensing Noise**: Observations regarding teammate states and user queues are corrupted by Gaussian noise ($\mathcal{N}(0, \sigma^2)$).

------

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy

### Setup

Bash

```
# Clone the repository
git clone https://github.com/YourUsername/UAV-MEC-Sim.git
cd UAV-MEC-Sim

# Install dependencies
pip install -r requirements.txt
```

------

## 🛠️ Usage

### 1. Training

Train the agents using the proposed **EVA-Gen (PRISM)** framework or baseline methods:

Bash

```
# Train on the default Urban map
python train.py --env-name UAV-MEC-Urban --algo eva_gen --seed 1

# Train under high occlusion (Severe Fragmentation)
python train.py --env-name UAV-MEC-Emergency --algo eva_gen --occlusion 0.5 --seed 1
```

### 2. Evaluation

Evaluate the trained models to obtain key system metrics (Success Rate, Violation Rate, CTC):

Bash

```
python eval.py --checkpoint ./models/eva_gen_urban.pt --num-episodes 100
```

------

## 📊 Comparison Methods

We include implementations of several state-of-the-art algorithms used as baselines in our experiments:

- **General MARL Methods**:
  - **MAPPO**: Multi-Agent PPO with centralized value function.
  - **MATD3 / MASAC**: Off-policy actor-critic methods adapted for multi-agent settings.
- **Domain-Specific Methods**:
  - **CMiMC**: Focuses on communication efficiency via mutual information.
  - **ExplabOff**: Enhances exploration for offloading decisions.

------

## 📈 Performance Metrics

The environment tracks specific metrics to validate system robustness and safety:

1. **CTC (Comprehensive Task Cost)**: The primary objective, aggregating delay, energy, and penalties.
2. **LS Violation Rate**: The ratio of critical tasks missing their deadlines (Key safety indicator).
3. **Success Rate**: The ratio of successfully completed tasks.
4. **System Throughput**: Total data processed per second.
5. **Load Balance**: Variance of computational load across UAVs.

------

## ⚙️ Configuration Parameters

Key simulation parameters can be adjusted in `config/env_config.yaml`:

YAML

```
env:
  num_uavs: 4
  num_users: 20
  area_size: 1000  # 1000m x 1000m
  
tasks:
  lambda_ls: 0.3  # Arrival rate for critical tasks
  lambda_lt: 0.7  # Arrival rate for background tasks
  penalty_weight: 50.0 # Penalty factor for LS violations

channel:
  occlusion_prob: 0.3 # Probability of NLoS (Fragmentation Level)
  noise_std: 0.1
```