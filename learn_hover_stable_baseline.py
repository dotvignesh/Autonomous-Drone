import os
import csv
import gymnasium as gym
from datetime import datetime
from gym_pybullet_drones.envs.HoverAviarySB import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



def make_env(rank):
    def _init():
        env = HoverAviary(obs=ObservationType("kin"), act=ActionType("rpm"))
        return Monitor(env)
    return _init

class LoggingCallback(BaseCallback):
    def __init__(self, log_path, verbose=1):
        super(LoggingCallback, self).__init__(verbose)
        self.log_path = log_path
        self.best_reward = float('-inf')

        with open(self.log_path, mode='w', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(["episode", "timestep", "reward"])

    def _on_step(self) -> bool:
        # Log episode metrics at the end of each episode
        for i, done in enumerate(self.locals["dones"]):
            if done:
                episode = self.num_timesteps // self.locals["env"].get_attr("CTRL_FREQ")[0]
                reward = self.locals["infos"][i].get("episode", {}).get("r", 0)
                timestep = self.num_timesteps
                if reward > self.best_reward:
                    self.best_reward = reward
                with open(self.log_path, mode='a', newline='') as log_file:
                    log_writer = csv.writer(log_file)
                    log_writer.writerow([episode, timestep, reward])

                if self.verbose > 0:
                    print(f"Episode: {episode} \t Timestep: {timestep} \t Reward: {reward:.2f} \t Best Reward: {self.best_reward:.2f}")
        return True


def train_hover():

    log_dir = "sb3_hover_logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "Train_Hover_LOG_hover.csv")

    num_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])


    check_env(make_env(0)())  

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        gamma=0.99,
        n_steps=2048,
        batch_size=1024,
        n_epochs=80,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard/"),
        seed=0
    )

 
    callback = LoggingCallback(log_path=log_file_path, verbose=1)
    model.learn(total_timesteps=int(3e6), callback=callback)

    model_path = os.path.join(log_dir, "ppo_hover_final")
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")
    env.close()


if __name__ == "__main__":
    train_hover()
