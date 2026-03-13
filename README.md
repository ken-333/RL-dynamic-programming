# RL Dynamic Programming — Policy Iteration & Value Iteration

> **EECE 5614 · Reinforcement Learning and Decision Making Under Uncertainty**
> Northeastern University · Spring 2026

Implementation of matrix-form **Policy Iteration** and **Value Iteration** applied to two MDP environments: a stochastic maze navigation task and a gene regulatory network control task.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Problem 1 — Stochastic Maze](#problem-1--stochastic-maze)
- [Problem 2 — Gene Regulatory Network](#problem-2--gene-regulatory-network)
- [Installation](#installation)
- [Usage](#usage)
- [Results Summary](#results-summary)

---

## Overview

This project implements two classic dynamic programming algorithms in **matrix/vector form**:

| Algorithm | Policy Evaluation | Convergence |
|---|---|---|
| **Policy Iteration** | Exact solve: `V = (I − γM(π))⁻¹ Rᵖ` | Policy unchanged |
| **Value Iteration** | Bellman backup: `V_new = max_a [R_sa + γ M[a] V]` | `max‖V_new − V‖ < θ` |

Both algorithms are applied to stochastic MDPs where transition probabilities are represented as explicit `N × N` matrices.

---

## Project Structure

```
project2/
├── problem1/               # Stochastic Maze MDP
│   ├── maze_env.py         # MDP builder: transition matrices M[a], reward R_sa
│   ├── policy_iteration.py # Matrix-form Policy Iteration
│   ├── value_iteration.py  # Matrix-form Value Iteration
│   ├── simulate.py         # Trajectory sampling under a given policy
│   ├── visualize.py        # Plotting: value function, policy arrows, paths
│   └── run_experiments.py  # Main entry point for all experiments
│
├── problem2/               # Gene Regulatory Network MDP
│   ├── gene_env.py         # MDP builder: p53-MDM2 network dynamics
│   ├── policy_iteration.py # Matrix-form Policy Iteration
│   ├── value_iteration.py  # Matrix-form Value Iteration
│   ├── simulate.py         # Episode simulation & AvgA computation
│   └── run_experiments.py  # Main entry point for all experiments
│
├── Project2.pdf            # Assignment specification
└── Project2_visualization_helper.ipynb  # Reference visualization notebook
```

---

## Problem 1 — Stochastic Maze

An agent navigates an **18 × 18 maze** (248 valid states) with walls, oil cells, and bump cells.

### MDP Specification

| Element | Details |
|---|---|
| State space | 248 cells (18×18 − 76 walls) |
| Actions | Up, Down, Left, Right |
| Transition | Main direction: `1−p`; each perpendicular: `p/2` |
| Rewards | −1 (step), −0.8 (wall hit), −5 (oil), −10 (bump), +200 (goal) |

### Experiments

**1 & 2 — PI vs VI across three scenarios:**

| Scenario | p | γ | θ |
|---|---|---|---|
| Base | 0.02 | 0.99 | 0.01 |
| Large Stochasticity | 0.40 | 0.99 | 0.01 |
| Small Discount | 0.02 | 0.40 | 0.01 |

**3 — Effect of stochasticity** (`p ∈ {0.02, 0.2, 0.6}`):
- Two independent sampled trajectories per `p` on the same figure
- Average cumulative reward over 10 trajectories (horizon T_max = 400)

**4 — Effect of bump penalty** (−10 vs −50):
- Compares optimal policy and path conservatism around hazards

### Run

```bash
cd problem1
python run_experiments.py
```

---

## Problem 2 — Gene Regulatory Network

Control of the **p53-MDM2 negative feedback loop**, a biological network with 4 genes (ATM, p53, WIP1, MDM2).

### MDP Specification

| Element | Details |
|---|---|
| State space | 16 binary states `{0,1}⁴` |
| Actions | 5: no-control + perturb one of 4 genes |
| Dynamics | `s_k = v̄(C s_{k−1}) ⊕ a_{k−1} ⊕ n_k` (XOR with Bernoulli noise) |
| Reward | `R(s,a,s') = 5‖s'‖₁ − ‖a‖₁` (maximize gene activation) |
| Goal | Keep all 4 genes ON (`s = [1,1,1,1]`) as long as possible |

### Experiments

| Part | Method | p_noise | Goal |
|---|---|---|---|
| (a) | Value Iteration | 0.045 | Optimal policy + compare AvgA vs no-control |
| (b) | Value Iteration | 0.18, 0.55 | Effect of noise on optimal policy and AvgA |
| (c) | Policy Iteration | 0.045 | Compare with VI result from (a) |

**AvgA** (average gene activation) is evaluated over 75 episodes × 150 steps:

$$\text{AvgA} = \frac{1}{75} \sum_{i=1}^{75} \frac{1}{150} \sum_{k=1}^{150} \|s_k\|_1 \in [0, 4]$$

### Run

```bash
cd problem2
python run_experiments.py
```

---

## Installation

```bash
pip install numpy matplotlib seaborn
```

Python 3.8+ required. No additional dependencies.

---

## Usage

Each `run_experiments.py` is self-contained. Running it executes all experiments sequentially and displays plots interactively.

```bash
# Problem 1: all scenarios + stochasticity + bump penalty analysis
python problem1/run_experiments.py

# Problem 2: VI and PI for gene network control
python problem2/run_experiments.py
```

---

## Results Summary

### Problem 1 — Key Observations

- **Large stochasticity** (p=0.4): paths become longer and more variable; V(start) decreases as random drift makes precise navigation harder.
- **Small discount** (γ=0.4): policy becomes "greedy" — shorter paths even at the cost of entering hazard cells, since distant rewards are heavily discounted.
- **PI vs VI**: both converge to the same optimal policy; PI requires fewer iterations but each step involves solving a linear system.

### Problem 2 — Key Observations

- Without control, the gene network converges to `[0,0,0,0]` (all OFF) — AvgA ≈ 0.
- Optimal control achieves AvgA close to 4, keeping the system near `[1,1,1,1]`.
- Higher noise (p=0.55) reduces controllability: AvgA drops as random bit-flips counteract the control signal.
- PI and VI converge to the same optimal policy at p=0.045.
