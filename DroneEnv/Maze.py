import numpy as np
import pybullet as p
from gymnasium import Env
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class Maze(BaseRLAviary):
    """Single agent RL problem: navigate a predefined environment."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=True,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        """Initialization of the predefined environment."""
        self.ENV_SIZE = 5  # The size of the environment
        self.EPISODE_LEN_SEC = 10
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.obstacle_ids = []  # Reset obstacle IDs
        self._addObstacles()
        return obs, info


    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, info

    
    def _addObstacles(self):
        """Add a complete environment using the bathroom.urdf file."""
        #super()._addObstacles()  # Call base obstacle method (if necessary)

        # Load the entire bathroom URDF
        self.model_id = p.loadURDF("my3dmodels/urdf/bedroom.urdf",
                   basePosition=[0, 0, 0],  # Center of the 5x5 space
                   baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                   globalScaling=1.0,  # Adjust scaling to fit within the 5x5 space
                   physicsClientId=self.CLIENT,
                   useFixedBase=True)

    def _computeReward(self):
        """Computes the reward based on the drone's position, stability, and efficiency."""
        reward = 0.0
        state = self._getDroneStateVector(0)
        pos = np.array([state[0], state[1], state[2]])

        goal_pos = np.array([self.ENV_SIZE / 2, self.ENV_SIZE / 2, 1.0])  # (x, y, z)
        distance_to_goal = np.linalg.norm(pos - goal_pos)


        distance_reward = max(0, 2 - distance_to_goal**2)
        reward += distance_reward


        roll_angle = abs(state[7])
        pitch_angle = abs(state[8])
        stability_penalty = roll_angle + pitch_angle
        stability_reward = max(0, 1 - stability_penalty)
        reward += stability_reward

        velocity = np.array([state[10], state[11], state[12]])
        goal_direction = (goal_pos - pos) / (np.linalg.norm(goal_pos - pos) + 1e-6)
        velocity_alignment = np.dot(velocity, goal_direction)
        direction_reward = max(0, velocity_alignment)
        reward += direction_reward

        velocity = np.linalg.norm(state[10:13])
        velocity_reward = min(velocity, 1)
        reward += velocity_reward

        # Check if reached the goal
        if distance_to_goal < 0.1:
            reward += 100  
            self.termination_reason = "Target Reached"

        # Penalty for being out of bounds
        if distance_to_goal > self.ENV_SIZE:
            reward -= 30

        # # Collision Avoidance Penalty
        # num_joints = p.getNumJoints(self.model_id)
        # collision_threshold = 0.1  # Minimum safe distance
        # for i in range(num_joints):
        #     link_state = p.getLinkState(self.model_id, i, physicsClientId=self.CLIENT)
        #     link_pos = np.array(link_state[0])  # Extract the position of the link
        #     distance_to_object = np.linalg.norm(pos - link_pos)
        #     if distance_to_object < collision_threshold:
        #         reward -= 10  # Penalize for being too close
        #         break  # No need to check further if penalty is applied

        return reward


    def _computeTerminated(self):
        """Check if the episode is terminated."""
        state = self._getDroneStateVector(0)
        pos = np.array([state[0], state[1]])
        goal_pos = np.array([self.ENV_SIZE / 2, self.ENV_SIZE / 2])

        # Terminate if the drone reaches the goal or time runs out
        if np.linalg.norm(pos - goal_pos) < 0.1:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
        if abs(state[0]) > 5 or abs(state[1]) > 5 or state[2] < 0 or state[2] > 3:
            self.termination_reason = "Out of Bounds"
            return True

        self.termination_reason = None
        return False

    def _computeTruncated(self):
        """Check if the episode is truncated."""
        return False  # No truncation logic

    def _computeInfo(self):
        """Compute and return environment-specific info."""
        return {"goal_position": [self.ENV_SIZE / 2, self.ENV_SIZE / 2]}

    def _clipAndNormalizeState(self, state):
        """Clip and normalize the drone's state."""
        MAX_POS = self.ENV_SIZE / 2
        MAX_VEL = 3.0
        clipped_pos = np.clip(state[0:3], -MAX_POS, MAX_POS)
        clipped_vel = np.clip(state[10:13], -MAX_VEL, MAX_VEL)

        normalized_pos = clipped_pos / MAX_POS
        normalized_vel = clipped_vel / MAX_VEL

        return np.hstack([normalized_pos, state[3:7], normalized_vel]).reshape(-1,)