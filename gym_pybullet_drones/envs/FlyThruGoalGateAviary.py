import os
import numpy as np
import pybullet as p
from importlib.resources import files

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.BaseNewRLAviary import ActionType, ObservationType, BaseNewRLAviary

class FlyThruGoalGateAviary(BaseNewRLAviary):
    """Single agent RL problem: fly through gates, with a specific goal gate."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=True,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        """Initialization of a single agent RL environment."""
        self.EPISODE_LEN_SEC = 8
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

    def _addObstacles(self):
        """Add obstacles to the environment."""
        super()._addObstacles()

        urdf_path = files('gym_pybullet_drones').joinpath('assets/architrave.urdf')

        # Non-goal gate
        p.loadURDF(str(urdf_path),
                [0, -1, 0.5],
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,  
                physicsClientId=self.CLIENT)

        # Goal gate (green color)
        goal_gate_id = p.loadURDF(str(urdf_path),
                                [0.5, -1.5, 0.5],
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True,  
                                physicsClientId=self.CLIENT)
        p.changeVisualShape(goal_gate_id, -1, rgbaColor=[0, 1, 0, 1])

        # Create pillars for both gates
        for i in range(8):  
            # Non-goal gate pillars
            p.loadURDF("cube_small.urdf",
                    [-0.25, -1, 0.05 + i * 0.05],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=True,  
                    physicsClientId=self.CLIENT)
            p.loadURDF("cube_small.urdf",
                    [0.25, -1, 0.05 + i * 0.05],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=True,  
                    physicsClientId=self.CLIENT)

            # Goal gate pillars
            p.loadURDF("cube_small.urdf",
                    [0.25, -1.5, 0.05 + i * 0.05],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=True,  
                    physicsClientId=self.CLIENT)
            p.loadURDF("cube_small.urdf",
                    [0.75, -1.5, 0.05 + i * 0.05],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=True,  
                    physicsClientId=self.CLIENT)


    def _computeReward(self):
        """Improved reward function to guide the drone to the goal gate."""
        state = self._getDroneStateVector(0)
        drone_position = state[0:3]
        drone_velocity = state[10:13]  
        goal_position = np.array([0.5, -1.5, 0.5])  # Goal gate position

        distance_to_goal = np.linalg.norm(drone_position - goal_position)
        distance_reward = 10 * (1 - np.tanh(2 * distance_to_goal))  

        direction_to_goal = (goal_position - drone_position) / (np.linalg.norm(goal_position - drone_position) + 1e-6)
        velocity_along_direction = np.dot(drone_velocity, direction_to_goal)
        directional_reward = max(0, 5 * velocity_along_direction) 

        # Gate passage rewards
        reward = 0
        if (0.25 <= drone_position[0] <= 0.75) and (-1.6 <= drone_position[1] <= -1.4) and (0.45 <= drone_position[2] <= 0.55):
            reward += 50  

        # Penalty for passing through the non-goal gate
        if (-0.25 <= drone_position[0] <= 0.25) and (-1.1 <= drone_position[1] <= -0.9) and (0.45 <= drone_position[2] <= 0.55):
            reward -= 40  

        orientation_penalty = abs(state[7]) + abs(state[8])
        stability_reward = max(0, 1 - orientation_penalty)  

        height_penalty = 0
        if drone_position[2] > 0.6:
            height_penalty = (drone_position[2] - 0.6) * 2
        elif drone_position[2] < 0.2:
            height_penalty = (0.2 - drone_position[2]) * 2

        # Total reward calculation
        reward += distance_reward  
        reward += directional_reward  
        reward += stability_reward  
        reward -= height_penalty  

        return reward


    def _computeTerminated(self):
        """Computes the current done value."""
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeTruncated(self):
        """Computes the current truncated value."""
        state = self._getDroneStateVector(0)

        # if abs(state[0]) > 2.0 or abs(state[1]) > 3.0 or state[2] > 1.2: 
        #     return True
        # if abs(state[7]) > 0.5 or abs(state[8]) > 0.5:
        #     return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False


    def _computeInfo(self):
        """Computes the current info dict(s)."""
        return {"answer": 42}

    def _clipAndNormalizeState(self,
                              state
                              ):
        """Normalizes a drone's state to the [-1,1] range."""
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi 

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                             clipped_pos_xy,
                                             clipped_pos_z,
                                             clipped_rp,
                                             clipped_vel_xy,
                                             clipped_vel_z
                                             )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi 
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                    normalized_pos_z,
                                    state[3:7],
                                    normalized_rp,
                                    normalized_y,
                                    normalized_vel_xy,
                                    normalized_vel_z,
                                    normalized_ang_vel,
                                    state[16:20]
                                    ]).reshape(20,)

        return norm_and_clipped
    
    def _clipAndNormalizeStateWarning(self,
                                     state,
                                     clipped_pos_xy,
                                     clipped_pos_z,
                                     clipped_rp,
                                     clipped_vel_xy,
                                     clipped_vel_z,
                                     ):
        """Debugging printouts associated to _clipAndNormalizeState."""
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGoalGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGoalGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGoalGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGoalGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGoalGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))