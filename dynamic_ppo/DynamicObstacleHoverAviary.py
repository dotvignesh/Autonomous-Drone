import numpy as np
import pybullet as p
from gymnasium import Env
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class DynamicObstacleHoverAviary(BaseRLAviary, Env):
    """HoverAviary with dynamic obstacles for Gymnasium."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 dynamic_obstacles=3,
                 obstacle_speed=0.05):
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_speed = obstacle_speed
        self.obstacle_ids = []
        self.TARGET_POS = np.array([0, 0, 1])  # Define target position
        self.EPISODE_LEN_SEC = 10  # Define the episode length in seconds
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obstacle_ids = []  # Reset obstacle IDs
        self._add_dynamic_obstacles()
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        self._move_dynamic_obstacles()
        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, info

    def _add_dynamic_obstacles(self):
        for _ in range(self.dynamic_obstacles):
            pos = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.5]
            obstacle = p.loadURDF("sphere2.urdf", pos)
            self.obstacle_ids.append((obstacle, pos))

    def _move_dynamic_obstacles(self):
        for i, (obstacle_id, pos) in enumerate(self.obstacle_ids):
            pos[0] += self.obstacle_speed * np.sin(self.step_counter / 20)
            pos[1] += self.obstacle_speed * np.cos(self.step_counter / 20)
            p.resetBasePositionAndOrientation(obstacle_id, pos, [0, 0, 0, 1])
            self.obstacle_ids[i] = (obstacle_id, pos)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        distance_to_target = np.linalg.norm(self.TARGET_POS - state[0:3])  # Distance to target position

        # Base reward: Encourage the drone to get closer to the target
        reward = max(0, 2 - distance_to_target**2)

        # Obstacle penalty: Penalize proximity to obstacles
        penalty = 0
        for _, pos in self.obstacle_ids:
            distance_to_obstacle = np.linalg.norm(state[0:3] - pos)
            penalty += max(0, 1 - distance_to_obstacle)  # Add penalty for being close to obstacles

        return reward - penalty

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)

        # Terminate if the drone reaches the target
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.1:  # Within 0.1m of the target
            self.termination_reason = "Target Reached"
            return True

        # Terminate if the drone is too far from the allowed area
        if abs(state[0]) > 5 or abs(state[1]) > 5 or state[2] < 0 or state[2] > 3:
            self.termination_reason = "Out of Bounds"
            return True

        # Terminate if the drone tilts too much
        if abs(state[7]) > 0.5 or abs(state[8]) > 0.5:  # Roll and pitch angles exceed limits
            self.termination_reason = "Excessive Tilt"
            return True

        # Terminate if the drone collides with an obstacle
        for _, pos in self.obstacle_ids:
            if np.linalg.norm(state[0:3] - pos) < 0.2:  # Collision threshold
                self.termination_reason = "Collision with Obstacle"
                return True

        # Otherwise, do not terminate
        self.termination_reason = None
        return False

    def _computeTruncated(self):
        # Truncate if the episode exceeds the time limit
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            self.truncation_reason = "Time Limit Exceeded"
            return True

        # Truncate if the drone is too far from the target area
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 10 or abs(state[1]) > 10 or state[2] > 5:
            self.truncation_reason = "Out of Allowed Area"
            return True

        # Otherwise, do not truncate
        self.truncation_reason = None
        return False

    def _computeInfo(self):
        return {
            "obstacles": len(self.obstacle_ids),
            "termination_reason": getattr(self, "termination_reason", None),
            "truncation_reason": getattr(self, "truncation_reason", None),
        }

