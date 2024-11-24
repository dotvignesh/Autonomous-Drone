import os
import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.BaseNewRLAviary import ActionType, ObservationType, BaseNewRLAviary

class CircuitAviary(BaseNewRLAviary):
    """Single agent RL problem: fly through gates in a circular circuit."""
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a circular circuit RL environment."""
        self.NUM_GATES = 8  # Number of gates in the circuit
        self.CIRCUIT_RADIUS = 2.0  # Radius of the circuit
        self.gates_positions = []  # Will store gate positions
        self.gates_passed = set()  # Track unique gates passed
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
        
        # Calculate gate positions in a circle
        for i in range(self.NUM_GATES):
            angle = (2 * np.pi * i) / self.NUM_GATES
            x = self.CIRCUIT_RADIUS * np.cos(angle)
            y = self.CIRCUIT_RADIUS * np.sin(angle)
            
            self.gates_positions.append([x, y])
            
            # Add gate (architrave)
            p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/architrave.urdf'),
                      [x, y, .55],
                      p.getQuaternionFromEuler([0, 0, angle]),
                      physicsClientId=self.CLIENT
                      )
            
            # Add pillars for each gate
            for h in range(10):
                # Right pillar
                p.loadURDF("cube_small.urdf",
                          [x + 0.3*np.cos(angle), y + 0.3*np.sin(angle), .02+h*0.05],
                          p.getQuaternionFromEuler([0, 0, 0]),
                          physicsClientId=self.CLIENT
                          )
                # Left pillar
                p.loadURDF("cube_small.urdf",
                          [x - 0.3*np.cos(angle), y - 0.3*np.sin(angle), .02+h*0.05],
                          p.getQuaternionFromEuler([0, 0, 0]),
                          physicsClientId=self.CLIENT
                          )

    def _computeReward(self):
        """Computes the current reward value."""
        reward = 0.0
        state = self._getDroneStateVector(0)
        pos = np.array([state[0], state[1]])
        
        # Calculate current position relative to circle center
        current_radius = np.sqrt(pos[0]**2 + pos[1]**2)
        current_angle = np.arctan2(pos[1], pos[0])
        if current_angle < 0:
            current_angle += 2 * np.pi
            
        # Reward for staying close to ideal circular path
        if abs(current_radius - self.CIRCUIT_RADIUS) < 0.5:
            reward += 1.0
            
        # Reward for moving in circular direction
        velocity = np.array([state[10], state[11]])
        expected_velocity = np.array([-np.sin(current_angle), np.cos(current_angle)])
        velocity_alignment = np.dot(velocity, expected_velocity)
        if velocity_alignment > 0:
            reward += 1.0 * velocity_alignment
        
        # Main reward for passing through gates
        for i, gate_pos in enumerate(self.gates_positions):
            gate_pos = np.array(gate_pos)
            dist_to_gate = np.linalg.norm(pos - gate_pos)
            
            if dist_to_gate < 0.3 and state[2] <= 0.5:  # Within gate bounds and correct height
                if i not in self.gates_passed:
                    self.gates_passed.add(i)
                    reward += 10
        
        # Height maintenance reward
        if abs(state[2] - 0.5) < 0.3:
            reward += 1.0
            
        # Bonus for completing the circuit
        if self._computeTruncated():
            reward += 100
            
        return reward

    def _computeTerminated(self):
        """Computes the current done value."""
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False
    
    def _computeTruncated(self):
        """Computes the current truncated value."""
        state = self._getDroneStateVector(0)
        pos = np.array([state[0], state[1]])
        
        # Check if near starting position after passing through gates
        if (np.linalg.norm(pos) < 0.3  # Near center
            and self.step_counter > self.PYB_FREQ * 5):  # Avoid immediate termination
            return True
        return False

    def _computeInfo(self):
        """Computes the current info dict(s)."""
        return {"answer": 42}

    def reset(self, seed=None, options=None):
        """Reset the environment and clear passed gates tracking."""
        self.gates_passed.clear()
        return super().reset(seed=seed, options=options)
        
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
        """Debugging printouts associated to `_clipAndNormalizeState`."""
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in CircuitAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in CircuitAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in CircuitAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in CircuitAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in CircuitAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))