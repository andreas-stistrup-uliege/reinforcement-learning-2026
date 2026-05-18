import os
import glob
from stable_baselines3 import SAC, PPO

CHECKPOINT_DIR = "checkpoints"

# Map name prefixes to SB3 classes
ALGO_MAP = {
    "sac": SAC,
    "ppo": PPO,
}

paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip")))

if not paths:
    print(f"No .zip files found in {CHECKPOINT_DIR}/")
else:
    print(f"Found {len(paths)} checkpoint(s).\n")

for path in paths:
    fname = os.path.basename(path).lower()
    algo_cls = next((cls for prefix, cls in ALGO_MAP.items() if fname.startswith(prefix)), None)

    if algo_cls is None:
        print(f"  [SKIP] {path}  — unknown prefix, not SAC or PPO")
        continue

    try:
        model = algo_cls.load(path)
        model.save(path.replace(".zip", ""))   # SB3 appends .zip automatically
        print(f"  [OK]   {path}")

        # Smoke-test: make sure .predict() works with a dummy observation
        obs_shape = model.observation_space.shape
        import numpy as np
        dummy_obs = model.observation_space.sample()
        action, info = model.predict(dummy_obs, deterministic=True)
        print(f"         predict OK — obs={obs_shape}  action={action.shape}")

    except Exception as e:
        print(f"  [FAIL] {path}")
        print(f"         {e}")

print("\nDone.")
