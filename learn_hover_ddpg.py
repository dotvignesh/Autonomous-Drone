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
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

class DroneTrainer:
    def __init__(self, max_training_timesteps=int(3e6),tau=0.001, gamma=0.99, lr_actor=0.000025, lr_critic=0.00025, hidden_dim1=64, hidden_dim2=128, memory_size=int(100000), batch_size=64):
        self.max_training_timesteps = max_training_timesteps
        self.tau = tau
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2  
        self.memory_size = int(memory_size)
        self.batch_size = batch_size
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

        self.random_seed = 0
        self.log_dir = "log_dir/"
        self.run_id = "hover_ddpg"
        self.checkpoint_base = os.path.join(self.log_dir, self.run_id)

        self.save_model_freq = int(1e5)

    def initialize_environment(self):
        self.env = HoverAviary(obs=self.obs_type, act=self.act_type)
        # self.episode_length = self.env.EPISODE_LEN_SEC * self.env.CTRL_FREQ
        # self.update_timestep = self.episode_length * 4
        # self.log_freq = self.episode_length * 2
        # self.print_freq = self.episode_length * 10

    def initialize_agent(self):
        self.agent = DDPG(
            self.state_dim,
            self.action_dim,
            self.lr_actor,
            self.lr_critic,
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

        self.log_file_path = os.path.join(self.log_dir, f"Train_Hover_LOG_{self.run_id}.csv")
        self.log_file = open(self.log_file_path, "w+")
        self.log_file.write("episode,timestep,reward\n")

        self.print_running_reward = 0
        self.print_running_episodes = 0
        self.log_running_reward = 0
        self.log_running_episodes = 0

        print(f"Logging initialized for run: {self.run_id}")
        print(f"Logging at: {self.log_file_path}")

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

    def save_model(self):
        checkpoint_path = os.path.join(
            self.checkpoint_base, f"{self.i_episode}_checkpoint_hover.pth"
        )
        print(f"Saving model at: {checkpoint_path}")
        self.agent.save(checkpoint_path)

    def train(self):
        np.random.seed(0)

        score_history = []
        for eps in tqdm(range(2)):
            obs,_ = self.env.reset()
            terminal = False
            score = 0
            i = 0
            while not terminal and i < 1000:
                # print("step: ", i+1)
                
                act = self.agent.select_action(obs)
                correct_dim_act = np.copy(act).reshape(1,-1)
                new_state, reward, terminal, truncated, info = self.env.step(correct_dim_act)
                self.agent.remember_transition(obs, act, reward, new_state, int(terminal))
                self.agent.learn()
                score += reward
                obs = new_state

                i = i + 1
            score_history.append(score)

        #if i % 25 == 0:
        #    agent.save_models()

        print('episode ', eps, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        
        self.env.close()


if __name__ == "__main__":
    trainer = DroneTrainer(max_training_timesteps=1e6, tau=1e-3, hidden_dim1=512, hidden_dim2=256, memory_size=1e5, batch_size=128)
    trainer.train()
