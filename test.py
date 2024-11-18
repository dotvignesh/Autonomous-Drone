import gymnasium as gym
from DynamicObstacleHoverAviary import DynamicObstacleHoverAviary

# Register the new environment
gym.envs.registration.register(
    id='DynamicObstacleHoverAviary-v0',
    entry_point='DynamicObstacleHoverAviary:DynamicObstacleHoverAviary',
    kwargs={'dynamic_obstacles': 3, 'obstacle_speed': 0.05}
)

# Instantiate and test the environment
if __name__ == "__main__":
    env = gym.make('DynamicObstacleHoverAviary-v0', gui=True)
    obs, info = env.reset()

    for episode in range(500):  # Run 5 episodes
        print(f"Starting Episode {episode + 1}")
        done, truncated = False, False
        while not (done or truncated):
            action = env.action_space.sample()  # Take a random action
            obs, reward, done, truncated, info = env.step(action)
            env.render()

            # Print step information for debugging
            print(f"Step Info: Reward={reward}, Done={done}, Truncated={truncated}")

        # Log termination or truncation reason
        print(f"Episode Ended: {info}")
        obs, info = env.reset()

    env.close()
