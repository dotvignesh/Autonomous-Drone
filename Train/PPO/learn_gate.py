import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from CustomRL.ppo import PPO
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

class DroneTrainer:
    def __init__(self):
        self.default_settings()
        self.initialize_environment()
        self.initialize_agent()
        self.setup_logging()
        self.best_reward = 0

    def default_settings(self):
        self.gui_enabled = True
        self.record_video = False
        self.output_folder = "results"
        self.colab_mode = False
        self.obs_type = ObservationType("kin")
        self.act_type = ActionType("rpm")
        self.state_dim = 12
        self.action_dim = 4
        
        # PPO Hyperparameters
        self.action_std = 0.6
        self.action_std_decay_rate = 0.05
        self.min_action_std = 0.1
        self.action_std_decay_freq = int(2.5e5)
        self.max_training_timesteps = int(3e6)
        self.ppo_epochs = 80
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.random_seed = 0
        self.log_dir = "log_dir/"
        self.run_id = "gates_new"
        self.checkpoint_base = os.path.join(self.log_dir, self.run_id)
        self.save_model_freq = int(1e5)

    def initialize_environment(self):
        self.env = FlyThruGateAviary(obs=self.obs_type, act=self.act_type)
        self.episode_length = self.env.EPISODE_LEN_SEC * self.env.CTRL_FREQ
        self.update_timestep = self.episode_length * 4
        self.log_freq = self.episode_length * 2
        self.print_freq = self.episode_length * 10

    def initialize_agent(self):
        self.agent = PPO(
            self.state_dim,
            self.action_dim,
            self.lr_actor,
            self.lr_critic,
            self.gamma,
            self.ppo_epochs,
            self.eps_clip,
            self.action_std,
        )

    def setup_logging(self):
        if not os.path.exists(self.checkpoint_base):
            os.makedirs(self.checkpoint_base)
        self.log_file_path = os.path.join(self.log_dir, f"Train_Gate_LOG.csv")
        self.log_file = open(self.log_file_path, "w+")
        self.log_file.write("episode,timestep,reward\n")
        self.print_running_reward = 0
        self.print_running_episodes = 0
        self.log_running_reward = 0
        self.log_running_episodes = 0

    def run_episode(self):
        obs, info = self.env.reset(seed=42, options={})
        current_ep_reward = 0
        
        for _ in range(self.episode_length):
            action = self.agent.select_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action[np.newaxis])
            done = terminated or truncated
            
            self.agent.buffer.rewards.append(reward)
            self.agent.buffer.is_terminals.append(done)
            current_ep_reward += reward
            
            self.handle_timestep_updates()
            if done:
                break
                
        return current_ep_reward

    def handle_timestep_updates(self):
        self.time_step += 1
        
        if self.time_step % self.update_timestep == 0:
            self.agent.update()
            
        if self.time_step % self.action_std_decay_freq == 0:
            self.agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)
            
        if self.time_step % self.log_freq == 0:
            avg_reward = round(self.log_running_reward / self.log_running_episodes, 4)
            self.log_file.write(f"{self.i_episode},{self.time_step},{avg_reward}\n")
            self.log_file.flush()
            self.log_running_reward = 0
            self.log_running_episodes = 0
            
        if self.time_step % self.print_freq == 0:
            avg_reward = round(self.print_running_reward / self.print_running_episodes, 2)
            print(f"Episode: {self.i_episode} \t Timestep: {self.time_step} \t Average Reward: {avg_reward}")

            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                best_model_path = os.path.join(self.checkpoint_base, "best_gate.pth")
                print(f"New best model found! Saving as: {best_model_path}")
                self.agent.save(best_model_path)

            self.print_running_reward = 0
            self.print_running_episodes = 0
            
        if self.time_step % self.save_model_freq == 0:
            self.save_model()

    def save_model(self):
        checkpoint_path = os.path.join(self.checkpoint_base, f"{self.i_episode}_checkpoint_gate.pth")
        print(f"Saving model at: {checkpoint_path}")
        self.agent.save(checkpoint_path)

    def train(self):
        self.time_step = 0
        self.i_episode = 0
        start_time = datetime.now().replace(microsecond=0)
        print(f"Started training at (GMT): {start_time}")

        while self.time_step <= self.max_training_timesteps:
            episode_reward = self.run_episode()
            self.print_running_reward += episode_reward
            self.print_running_episodes += 1
            self.log_running_reward += episode_reward
            self.log_running_episodes += 1
            self.i_episode += 1

        end_time = datetime.now().replace(microsecond=0)
        print(f"Finished training at (GMT): {end_time}")
        print(f"Total training time: {end_time - start_time}")
        self.log_file.close()
        self.env.close()

if __name__ == "__main__":
    trainer = DroneTrainer()
    trainer.train()