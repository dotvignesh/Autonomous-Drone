import os
import csv
from gym_pybullet_drones.envs.HoverAviarySB import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

class LoggingCallback(BaseCallback):
    def __init__(self, log_path, verbose=1):
        super(LoggingCallback, self).__init__(verbose)
        self.log_path = log_path
        self.best_reward = float('-inf')

        with open(self.log_path, mode='w', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(["episode", "timestep", "reward"])

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                episode = self.num_timesteps // self.locals["env"].envs[0].CTRL_FREQ 
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


def train_hover_ddpg():
    log_dir = "sb3_hover_logs_ddpg/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "Train_Hover_LOG_ddpg.csv")

    env = Monitor(HoverAviary(obs=ObservationType("kin"), act=ActionType("rpm")))

    check_env(env) 
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.001,
        buffer_size=int(1e6),
        learning_starts=10000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"), 
        gradient_steps=50,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard/"),
        seed=0
    )

    callback = LoggingCallback(log_path=log_file_path, verbose=1)
    model.learn(total_timesteps=int(3e6), callback=callback)

    model_path = os.path.join(log_dir, "ddpg_hover_final")
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")
    env.close()

if __name__ == "__main__":
    train_hover_ddpg()
