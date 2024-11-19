import os
import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.BaseNewRLAviary import ActionType, ObservationType, BaseNewRLAviary

class FlyThruGateAviary(BaseNewRLAviary):
    """Single agent RL problem: fly through a gate."""
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=True,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
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
        # First gate - lowered height to 0.4
        p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/architrave.urdf'),
                   [0, -1, .4],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        # Second gate - lowered height to 0.4
        p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/architrave.urdf'),
                   [0, -2, .4],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        
        # Create pillars for both gates
        for i in range(8):  # Reduced height of pillars
            # First gate pillars
            p.loadURDF("cube_small.urdf",
                       [-.3, -1, .02+i*0.05],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [.3, -1, .02+i*0.05],
                       p.getQuaternionFromEuler([0,0,0]),
                       physicsClientId=self.CLIENT
                       )
            # Second gate pillars
            p.loadURDF("cube_small.urdf",
                       [-.3, -2, .02+i*0.05],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [.3, -2, .02+i*0.05],
                       p.getQuaternionFromEuler([0,0,0]),
                       physicsClientId=self.CLIENT
                       )
            
            # Add walls between gates
            if i < 6:  # Reduced height of walls
                p.loadURDF("cube_small.urdf",
                          [0.15, -1.5, .02+i*0.05],
                          p.getQuaternionFromEuler([0, 0, 0]),
                          physicsClientId=self.CLIENT
                          )
                p.loadURDF("cube_small.urdf",
                          [-0.15, -1.5, .02+i*0.05],
                          p.getQuaternionFromEuler([0, 0, 0]),
                          physicsClientId=self.CLIENT
                          )

    def _computeReward(self):
        """Computes the current reward value."""
        state = self._getDroneStateVector(0)
        norm_ep_time = (self.step_counter/self.PYB_FREQ) / self.EPISODE_LEN_SEC
        
        # Target position with lower height (0.35 instead of 0.75)
        target_pos = np.array([0, -2*norm_ep_time, 0.35])
        distance_reward = max(0, 1 - np.linalg.norm(target_pos - state[0:3]))
        
        # Height penalty - discourage flying too high
        height_penalty = max(0, state[2] - 0.5) * 0.5  # Penalty increases with height above 0.6m
        
        # Additional rewards for passing through gates
        bonus_reward = 0
        # First gate check - adjusted height check
        if (state[0] >= -.27) and (state[0] <= .27) and (state[1] >= -1.1) and (state[1] <= -0.9) and (state[2] <= 0.45):
            bonus_reward += 5
        # Second gate check - adjusted height check
        if (state[0] >= -.27) and (state[0] <= .27) and (state[1] >= -2.1) and (state[1] <= -1.9) and (state[2] <= 0.45):
            bonus_reward += 10
            
        return distance_reward + bonus_reward - height_penalty

    def _computeTerminated(self):
        """Computes the current done value."""
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False
        
    def _computeTruncated(self):
        """Computes the current truncated value."""
        state = self._getDroneStateVector(0)
        
        # Position limits - reduced height limit
        if (abs(state[0]) > 1.5 or    # X limit
            abs(state[1]) > 2.5 or     # Y limit
            state[2] > 1.0):           # Z limit reduced from 2.0 to 1.0
            return True
            
        # Tilt limits
        if (abs(state[7]) > 0.4 or     # Roll limit
            abs(state[8]) > 0.4):      # Pitch limit
            return True
            
        # Time limit
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
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

        MAX_PITCH_ROLL = np.pi # Full range

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
        normalized_y = state[9] / np.pi # No reason to clip
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
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))