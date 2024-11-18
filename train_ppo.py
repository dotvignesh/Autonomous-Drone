import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os
import torch
import torch_directml
gpu_device = torch_directml.device()

from DynamicObstacleHoverAviary import DynamicObstacleHoverAviary


gym.envs.registration.register(
    id='DynamicObstacleHoverAviary-v0',
    entry_point='DynamicObstacleHoverAviary:DynamicObstacleHoverAviary',
    kwargs={'dynamic_obstacles': 3, 'obstacle_speed': 0.05}
)


models_dir = "models/PPO"
os.makedirs(models_dir, exist_ok=True)

# Create the environment
env = gym.make('DynamicObstacleHoverAviary-v0', gui=False)
env = Monitor(env) 

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device=gpu_device
)


total_timesteps = 50000
model.learn(total_timesteps=total_timesteps)

model_path = os.path.join(models_dir, "ppo_dynamic_obstacles")
model.save(model_path)

env.close()

env = gym.make('DynamicObstacleHoverAviary-v0', gui=False)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

env.close()
