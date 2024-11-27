import os
import gymnasium as gym
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviarySB import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import time

def test_hover():
    model_path = "sb3_hover_logs/ppo_hover_final.zip" 
    if not os.path.exists(model_path):
        print("Model file not found! Please check the path.")
        return


    model = PPO.load(model_path)

    env = HoverAviary(
        obs=ObservationType("kin"),
        act=ActionType("rpm")
    )


    obs, info = env.reset(seed=42)

    for episode in range(5):

        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0

        while not done:

            action, _states = model.predict(obs, deterministic=True)


            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env.render()
            time.sleep(0.1)

            done = terminated or truncated

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    test_hover()
