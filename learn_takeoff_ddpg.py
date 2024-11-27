import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from CustomRL.ddpg import DDPG

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

class DroneTrainer:
    def __init__(self, max_training_timesteps=int(3e6), tau=0.001, gamma=0.99, lr_actor=0.000025, lr_critic=0.00025, noise_decay = None, hidden_dim1=128, hidden_dim2=64, memory_size=int(100000), batch_size=64):
        self.max_training_timesteps = max_training_timesteps
        self.tau = tau
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2  
        self.memory_size = int(memory_size)
        self.batch_size = batch_size
        self.noise_decay = noise_decay
        self.default_settings()
        self.initialize_environment()
        self.initialize_agent()
        self.setup_logging()
        self.best_reward = 0
        np.random.seed(0)

    def default_settings(self):
        self.gui_enabled = True
        self.record_video = False
        self.output_folder = "results"
        self.colab_mode = False

        self.obs_type = ObservationType("kin")
        self.act_type = ActionType("rpm")

        self.state_dim = 12
        self.action_dim = 4

        self.random_seed = 0
        self.log_dir = "log_dir/"
        self.run_id = "takeoff_ddpg"
        self.checkpoint_base = os.path.join(self.log_dir, self.run_id)

        self.save_model_freq = int(1e5)

    def initialize_environment(self):
        self.env = TakeoffAviary(obs=self.obs_type, act=self.act_type)
        self.episode_length = self.env.EPISODE_LEN_SEC * self.env.CTRL_FREQ
        self.update_timestep = self.episode_length * 4
        self.log_freq = self.episode_length * 2
        self.print_freq = self.episode_length * 10

    def initialize_agent(self):
        self.agent = DDPG(
            self.state_dim,
            self.action_dim,
            self.lr_actor,
            self.lr_critic,
            self.noise_decay,
            self.gamma,
            self.tau,
            self.hidden_dim1,
            self.hidden_dim2,
            self.memory_size,
            self.batch_size
        )

    def setup_logging(self):
        if not os.path.exists(self.checkpoint_base):
            os.makedirs(self.checkpoint_base)

        self.log_file_path = os.path.join(self.log_dir, f"Train_Takeoff_LOG_{self.run_id}.csv")
        self.log_file = open(self.log_file_path, "w+")
        self.log_file.write("episode,timestep,reward\n")

        self.print_running_reward = 0
        self.print_running_episodes = 0
        self.log_running_reward = 0
        self.log_running_episodes = 0

        print(f"Logging initialized for run: {self.run_id}")
        print(f"Logging at: {self.log_file_path}")
    
    def handle_timestep_updates(self):
        self.time_step += 1

        if self.time_step % self.log_freq == 0:
            avg_reward = round(self.log_running_reward / self.log_running_episodes, 4)
            self.log_file.write(f"{self.i_episode},{self.time_step},{avg_reward}\n")
            self.log_file.flush()
            self.log_running_reward = 0
            self.log_running_episodes = 0

        if self.time_step % self.print_freq == 0:
            avg_reward = round(self.print_running_reward / self.print_running_episodes, 2)
            print(
                f"Episode: {self.i_episode} \t Timestep: {self.time_step} \t Average Reward: {avg_reward}"
            )

            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                best_model_path = os.path.join(self.checkpoint_base, "best_takeoff_ddpg.pth")
                print(f"New best model found! Saving as: {best_model_path}")
                self.agent.save(best_model_path)

            self.print_running_reward = 0
            self.print_running_episodes = 0

        if self.time_step % self.save_model_freq == 0:
            self.save_model()

    def save_model(self):
        checkpoint_path = os.path.join(
            self.checkpoint_base, f"{self.i_episode}_checkpoint_takeoff_ddpg.pth"
        )
        print(f"Saving model at: {checkpoint_path}")
        self.agent.save(checkpoint_path)

    def run_episode(self):
        obs, info = self.env.reset()
        current_ep_reward = 0

        for _ in range(self.episode_length):
            action = self.agent.select_action(obs)
            corrected_dim_act = np.copy(action).reshape(1, -1)
            new_state, reward, terminated, truncated, info = self.env.step(corrected_dim_act)
            done = terminated or truncated

            self.agent.remember_transition(obs, action, reward, new_state, int(done))
            self.agent.learn()
            current_ep_reward += reward
            self.handle_timestep_updates()
            
            if done:
                break
        
        self.agent.reset_OU_noise()
        self.agent.decay_OU_noise()
        return current_ep_reward
    
    def train(self):
        self.time_step = 0
        self.i_episode = 0
        start_time = datetime.now().replace(microsecond=0)
        print(f"Started training at (GMT): {start_time}")
        with tqdm(total=self.max_training_timesteps) as pbar:
            while self.time_step <= self.max_training_timesteps:
                episode_reward = self.run_episode()
                self.print_running_reward += episode_reward
                self.print_running_episodes += 1
                self.log_running_reward += episode_reward
                self.log_running_episodes += 1
                self.i_episode += 1
                pbar.update(self.time_step - pbar.n)

        end_time = datetime.now().replace(microsecond=0)
        print(f"Finished training at (GMT): {end_time}")
        print(f"Total training time: {end_time - start_time}")
        self.log_file.close()
        self.env.close()


if __name__ == "__main__":
    trainer = DroneTrainer(max_training_timesteps=int(5e5), tau=1e-3, gamma=0.99, lr_actor=0.00005, lr_critic=0.0005, noise_decay=0.9999, hidden_dim1=512, hidden_dim2=512, memory_size=int(1e6), batch_size=128)
    trainer.train()
