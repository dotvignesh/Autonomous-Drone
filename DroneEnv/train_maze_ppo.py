import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from Maze import Maze
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os
import torch

gpu_device = torch.device("cuda:0")
gym.envs.registration.register(
    id='MazeObs-v0',
    entry_point='Maze:Maze',
)

env = gym.make('MazeObs-v0', gui=False)
env = Monitor(env) 

# Create the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=100,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    tensorboard_log="./ppo_tensorboard/"
)

# Train the model
model.learn(total_timesteps=1000000)

# Save the trained model
model.save("ppo_maze_drone")

# # Test the trained model
# obs, _ = env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, _ = env.reset()

env.close()

env = gym.make('MazeObs-v0', gui=False)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

env.close()
