import os
import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from BaseNewRLAviary import ActionType, ObservationType, BaseNewRLAviary


class CircuitAviary(BaseNewRLAviary):
    """Single agent RL problem: fly through gates in a circular circuit."""
    
    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=True,
                 record=True,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a circular circuit RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.NUM_GATES = 8  # Number of gates in the circuit
        self.CIRCUIT_RADIUS = 2.0  # Radius of the circuit
        self.gates_positions = []  # Will store gate positions
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

    ################################################################################
    
    def _addObstacles(self):
        """Add obstacles to the environment.
        Creates a circular arrangement of gates oriented tangentially to form a circuit.
        """
        super()._addObstacles()
        
        # Calculate gate positions in a circle
        for i in range(self.NUM_GATES):
            angle = (2 * np.pi * i) / self.NUM_GATES
            x = self.CIRCUIT_RADIUS * np.cos(angle)
            y = self.CIRCUIT_RADIUS * np.sin(angle)
            
            # Store gate position for reward calculation
            self.gates_positions.append([x, y])
            
            # Add gate (architrave) - oriented tangentially to the circle
            p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/architrave.urdf'),
                      [x, y, .55],
                      p.getQuaternionFromEuler([0, 0, angle]),
                      physicsClientId=self.CLIENT
                      )
            
            # Add pillars for each gate - perpendicular to the tangent line
            for h in range(10):
                # Right pillar - angle points towards the center
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

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value based on passing through gates."""
        reward = 0.0
        state = self._getDroneStateVector(0)
        pos = np.array([state[0], state[1]])
        
        # Check passing through each gate
        for gate_pos in self.gates_positions:
            gate_pos = np.array(gate_pos)
            dist_to_gate = np.linalg.norm(pos - gate_pos)
            
            if dist_to_gate < 0.3 and state[2] <= 0.5:  # Within gate bounds and correct height
                reward += 10
        
        # Add distance-based penalty to encourage movement
        closest_gate_dist = min([np.linalg.norm(pos - np.array(gate_pos)) for gate_pos in self.gates_positions])
        reward += -2 * closest_gate_dist  # Penalty based on distance to nearest gate
        
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).
        Episode is truncated when drone completes a full circuit.
        """
        state = self._getDroneStateVector(0)
        pos = np.array([state[0], state[1]])
        
        # Check if near starting position after passing through gates
        if (np.linalg.norm(pos) < 0.3  # Near center
            and self.step_counter > self.PYB_FREQ * 5):  # Avoid immediate termination
            return True
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
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
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
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