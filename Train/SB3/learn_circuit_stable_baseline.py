import os
import csv
from datetime import datetime
from gymnasium import spaces
from gym_pybullet_drones.envs.CircuitAviary import CircuitAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank):
    def _init():
        env = CircuitAviary(obs=ObservationType("kin"), act=ActionType("rpm"), gui=False)
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

def train_circuit():
    log_dir = "sb3_circuit_logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "Train_Circuit_LOG.csv")

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
    model.learn(total_timesteps=int(1e6), callback=callback)

    model_path = os.path.join(log_dir, "ppo_circuit_final")
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")
    env.close()

if __name__ == "__main__":
    train_circuit()

# import os
# import csv
# from datetime import datetime
# from gymnasium import spaces
# from gym_pybullet_drones.envs.CircuitAviary import CircuitAviary
# from gym_pybullet_drones.utils.enums import ObservationType, ActionType
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.vec_env import SubprocVecEnv

# def make_env(rank):
#     def _init():
#         env = CircuitAviary(obs=ObservationType("kin"), act=ActionType("rpm"), gui=False)
#         return Monitor(env)
#     return _init

# class LoggingCallback(BaseCallback):
#     def __init__(self, log_path, model_save_path, verbose=1):
#         super(LoggingCallback, self).__init__(verbose)
#         self.log_path = log_path
#         self.model_save_path = model_save_path
#         self.best_reward = float('-inf')

#         # Initialize the log file
#         with open(self.log_path, mode='w', newline='') as log_file:
#             log_writer = csv.writer(log_file)
#             log_writer.writerow(["episode", "timestep", "reward"])

#     def _on_step(self) -> bool:
#         for i, done in enumerate(self.locals["dones"]):
#             if done:
#                 # Calculate the episode number
#                 episode = self.num_timesteps // self.locals["env"].get_attr("CTRL_FREQ")[0]
#                 # Get the reward from the info dict
#                 reward = self.locals["infos"][i].get("episode", {}).get("r", 0)
#                 timestep = self.num_timesteps
#                 if reward > self.best_reward:
#                     self.best_reward = reward
#                     # Save the model
#                     self.model.save(os.path.join(self.model_save_path, "best_model.zip"))

#                 # Log the episode data
#                 with open(self.log_path, mode='a', newline='') as log_file:
#                     log_writer = csv.writer(log_file)
#                     log_writer.writerow([episode, timestep, reward])

#                 if self.verbose > 0:
#                     print(f"Episode: {episode} \t Timestep: {timestep} \t Reward: {reward:.2f} \t Best Reward: {self.best_reward:.2f}")
#         return True

# def train_circuit():
#     log_dir = "sb3_circuit_logs/"
#     os.makedirs(log_dir, exist_ok=True)
#     log_file_path = os.path.join(log_dir, "Train_Circuit_LOG.csv")
#     model_save_path = os.path.join(log_dir, "best_model")
#     os.makedirs(model_save_path, exist_ok=True)

#     num_envs = 8
#     env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

#     check_env(make_env(0)())  

#     model = PPO(
#         policy="MlpPolicy",
#         env=env,
#         learning_rate=0.0003,
#         gamma=0.99,
#         n_steps=2048,
#         batch_size=1024,
#         n_epochs=80,
#         clip_range=0.2,
#         gae_lambda=0.95,
#         vf_coef=0.5,
#         ent_coef=0.01,
#         verbose=1,
#         tensorboard_log=os.path.join(log_dir, "tensorboard/"),
#         seed=0
#     )

#     callback = LoggingCallback(log_path=log_file_path, model_save_path=model_save_path, verbose=1)
#     model.learn(total_timesteps=int(5e5), callback=callback)

#     env.close()

# if __name__ == "__main__":
#     train_circuit()
