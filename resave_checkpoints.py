from stable_baselines3 import PPO
PPO.load("checkpoints\ppo_dogfight.zip").save("checkpoints\ppo_dogfight_pinned.zip")

from stable_baselines3 import SAC
SAC.load("checkpoints\sac_dogfight.zip").save("checkpoints\sac_dogfight_pinned.zip")
