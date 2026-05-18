from stable_baselines3 import SAC

def load_model(path=None):
    return SAC.load(path or "checkpoints/sac_dogfight.zip")