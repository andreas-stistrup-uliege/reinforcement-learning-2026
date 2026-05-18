# INFO8003 — Reinforcement Learning Project

Comparing **SAC** and **PPO** (via Stable-Baselines3) across three UAV environments from [PyFlyt](https://github.com/jjshoots/PyFlyt): hover stabilisation, waypoint navigation with a curriculum, and fixed-wing dogfighting with self-play.

All runs are logged to [WandB](https://wandb.ai/info8003-rl-project).

---

## Repository structure

```
.
├── hover_run.py          # Hover experiment (SAC + PPO, 5 flight modes)
├── waypoints_run.py      # Waypoints curriculum experiment (SAC + PPO, 5 flight modes)
├── dogfight_run.py       # Dogfight self-play experiment (SAC + PPO)
├── resave_checkpoints.py # Loads and saves models to ensure compatibility
├── env_probing.ipynb     # Environment exploration & MDP formalization
├── scripts/              # Course given scripts
├── checkpoints/          # Saved .zip model files (created at runtime)
└── logs/                 # Training curve PNGs (created at runtime) & terminal logs
```

---


## Environments

All three environments use PyFlyt's quadrotor physics. Five flight modes are tested across all scripts:

| Flight mode | Description |
|---|---|
| `7` | Fully assisted (attitude + altitude hold) |
| `6` | Attitude hold only |
| `4` | Angular velocity control |
| `0` | Velocity control |
| `-1` | Full manual (raw motor commands) |

Higher (more negative) flight mode numbers correspond to harder, lower-level control.

---

## Experiments

### Hover (`hover_run.py`)

Trains SAC and PPO to hold a fixed position for up to 500 steps. Runs 150,000 timesteps per algorithm per flight mode.

```bash
python hover_run.py
```

Checkpoints saved to `checkpoints/sac_hover_mode{N}.zip` and `checkpoints/ppo_hover_mode{N}.zip`.

---

### Waypoints (`waypoints_run.py`)

Trains SAC and PPO on a progressive curriculum: the agent must first reach 1 waypoint, then 2, 3, and finally 4. The environment advances to the next stage once a 70% success rate is achieved over a 10-episode probe.

Each curriculum stage has a fixed budget of 250,000 timesteps (1,000,000 total maximum).

```bash
python waypoints_run.py
```

To change which flight modes are trained, edit the `MODES` list at the top of the script:

```python
MODES = [7, 6, 4, 0, -1]
```

Checkpoints saved to `checkpoints/{sac,ppo}_waypoints_mode{N}.zip`.

---

### Dogfight (`dogfight_run.py`)

Trains SAC and PPO on a fixed-wing dogfight task using **self-play with snapshot opponents**. At regular intervals, a frozen copy of the current policy is added to a pool (max size 5) and the active opponent is sampled uniformly from the pool. This prevents overfitting to any single opponent style.

A small dense reward shaping term rewards closing distance and heading toward the opponent.

Runs for 900,000 timesteps per algorithm.

```bash
python dogfight_run.py
```

Checkpoints saved to `checkpoints/sac_dogfight.zip` and `checkpoints/ppo_dogfight.zip`.

---

## Environment exploration (`env_probing.ipynb`)

A Jupyter notebook that systematically probes all three environments before training. Run it first to understand the MDP structure.

The notebook covers:

| Section | What is measured |
|---|---|
| **1. Spaces** | Observation and action space shape, dtype, and bounds per flight mode |
| **2. Observation anatomy** | Per-component meaning, range, and variance under a random policy |
| **3. Action space by flight mode** | How bounds and semantics change across modes |
| **4. Reward signal** | Distribution, density, range, coefficient of variation per step |
| **5. Episode dynamics** | Length distribution, early termination rate |
| **6. Transition dynamics** | State-change magnitude `‖s′−s‖₂`, per-dimension volatility |
| **7. Random baseline returns** | All modes × all environments — the floor every trained agent must beat |
| **8. MDP formalization** | Full `⟨S, A, P, R, γ⟩` for Hover, Waypoints, and Dogfight |
| **9. Summary comparison** | Side-by-side table across all three environments |

Key findings documented in the notebook:

- **Hover** — `s ∈ ℝ²¹`, dense reward, deterministic transitions, difficulty scales strongly with flight mode.
- **Waypoints** — `s ∈ ℝ³³` (Dict obs flattened by `FlattenWaypointEnv`), sparse + shaped reward, deterministic transitions.
- **Dogfight** — `s ∈ ℝ³⁷`, sparse combat reward (high zero%), effectively stochastic transitions due to the opponent policy.


---

## Hyperparameters

Both algorithms use `lr = 3e-4` and `γ = 0.99` across all experiments. Differences are task-specific:

| Parameter | SAC | PPO |
|---|---|---|
| Entropy coeff `α` | 0.2 (0.1 dogfight) | — |
| Soft update `τ` | 0.005 | — |
| Replay buffer | 500,000 | — |
| Batch size | 256 (hover/waypoints) | 64 (hover/waypoints), 128 (dogfight) |
| Rollout steps | — | 2048 (hover/waypoints), 4096 (dogfight) |
| Epochs per update | — | 10 |
| Clip range `ε` | — | 0.2 |
| GAE `λ` | — | 0.95 |

---

## Reproducibility

All scripts fix `SEED = 42` for NumPy and SB3.