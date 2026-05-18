from stable_baselines3 import PPO

def load_model(path=None):
    return PPO.load(path or "checkpoints/ppo_dogfight.zip")