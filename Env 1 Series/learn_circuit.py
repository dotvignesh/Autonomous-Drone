"""Script demonstrating the use of `gym_pybullet_drones`' Gymnasium interface.

Class HoverAviary is used as a learning env for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3`.

"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.utils.Logger import Logger
from CircuitAviary import CircuitAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gymnasium.envs.registration import register

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

register(
    id="circuit-aviary-v0",
    entry_point="CircuitAviary:CircuitAviary",
)

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=False, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO):

    #### Check the environment's spaces ########################
    env = gym.make("circuit-aviary-v0")
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    #### Train the model #######################################    
    
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                tensorboard_log="./tensor_result/algo_circuit_tensorboard/"
                )
           
    model.learn(total_timesteps=2e6, progress_bar=True) # Typically not enough
    
    #### Save the model ########################################
    model.save('success_model.zip')
    
    #### Show (and record a video of) the model's performance ##
    env = CircuitAviary(gui=gui,
                        record=record_video
                        )
    
    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )
    obs, info = env.reset(seed=42, options={})
    start = time.time()
    for i in range(3*env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = env.step(action)
        logger.log(drone=0,
                   timestamp=i/env.CTRL_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                   control=np.zeros(12)
                   )
        env.render()
        print(terminated)
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            obs = env.reset(seed=42, options={})
    env.close()

    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using HoverAviary')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))