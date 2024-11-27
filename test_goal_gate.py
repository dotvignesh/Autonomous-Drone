import os
import time
from datetime import datetime
import numpy as np
from CustomRL.ppo import PPO
from gym_pybullet_drones.envs.FlyThruGoalGateAviary import FlyThruGoalGateAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Constants for default settings
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False 
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS = ObservationType('kin')  # Options: 'kin', 'rgb'
DEFAULT_ACT = ActionType('rpm')       # Options: 'rpm', 'pid', 'vel', 'one_d_rpm', 'one_d_pid'

# PPO configuration constants
STATE_DIM = 12
ACTION_DIM = 4
ACTION_STD = 0.1
ACTOR_LR = 0.0003
CRITIC_LR = 0.001
GAMMA = 0.99
K_EPOCHS = 80
EPS_CLIP = 0.2
CHECKPOINT_PATH = "log_dir/goal_training/best_goal.pth"

# Test environment configuration  
TEST_EPISODE_EXTRA_TIME = 20  # Extra seconds added to EPISODE_LEN_SEC for testing

def setup_environment(output_folder):
    """Create the FlyThruGate environment and ensure output folder exists."""
    output_dir = os.path.join(output_folder, f'recording_{datetime.now().strftime("%m.%d.%Y_%H.%M.%S")}')
    os.makedirs(output_dir, exist_ok=True)
    return FlyThruGoalGateAviary(gui=DEFAULT_GUI, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=DEFAULT_RECORD_VIDEO)

def load_agent(checkpoint_path):
    """Initialize and load a PPO agent from the given checkpoint path."""
    agent = PPO(STATE_DIM, ACTION_DIM, ACTOR_LR, CRITIC_LR, GAMMA, K_EPOCHS, EPS_CLIP, ACTION_STD)
    print(f"Loading trained model from: {checkpoint_path}")
    agent.load(checkpoint_path)

    agent.save("tested_goal_gate.pth")
    return agent

def test_episode(env, agent, max_steps, render=True):
    """Run a single test episode and return the cumulative reward."""
    obs, _ = env.reset(seed=42, options={})
    cumulative_reward = 0
    start_time = datetime.now().replace(microsecond=0)
    frame_start = time.time()

    frame_start = time.time()
    frame_duration = 1/8

    for step in range(max_steps):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(np.expand_dims(action, axis=0))
        cumulative_reward += reward

        if render:
            env.render()
            sync(step, frame_start, frame_duration)

        if terminated or truncated:
            break

    print(f"Episode completed with total reward: {round(cumulative_reward, 2)}")
    return cumulative_reward

def test():
    """Main testing function."""
    print("===== Starting Test =====")

    # Initialize environment and agent
    env = setup_environment(DEFAULT_OUTPUT_FOLDER) 
    agent = load_agent(CHECKPOINT_PATH)

    # Calculate max steps for the test
    max_steps = int((env.EPISODE_LEN_SEC + TEST_EPISODE_EXTRA_TIME) * env.CTRL_FREQ)

    # Run a test episode
    total_reward = test_episode(env, agent, max_steps)

    print("===== Test Complete =====")
    print(f"Total reward from test: {round(total_reward, 2)}")

    env.close()

if __name__ == '__main__':
    test()