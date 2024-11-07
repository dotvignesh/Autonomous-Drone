import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from BaseNewRLAviary import ActionType, ObservationType, BaseNewRLAviary

class MazeAviary(BaseNewRLAviary):

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=True,
                 record=True,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
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

        super()._addObstacles()


        wall_thickness = 0.1
        wall_height = 1.0
        self.maze_size = 5.0  



        p.loadURDF("cube_long.urdf",
                [0, self.maze_size / 2, wall_height / 2],
                p.getQuaternionFromEuler([0, 0, 0]),
                globalScaling=self.maze_size,
                physicsClientId=self.CLIENT
                )

        p.loadURDF("cube_long.urdf",
                [0, -self.maze_size / 2, wall_height / 2],
                p.getQuaternionFromEuler([0, 0, 0]),
                globalScaling=self.maze_size,
                physicsClientId=self.CLIENT
                )

        p.loadURDF("cube_long.urdf",
                [-self.maze_size / 2, 0, wall_height / 2],
                p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                globalScaling=self.maze_size,
                physicsClientId=self.CLIENT
                )

        p.loadURDF("cube_long.urdf",
                [self.maze_size / 2 - 1.0, 0, wall_height / 2],
                p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                globalScaling=self.maze_size - 2.0,
                physicsClientId=self.CLIENT
                )


        p.loadURDF("cube_long.urdf",
                [0, self.maze_size / 4, wall_height / 2],
                p.getQuaternionFromEuler([0, 0, 0]),
                globalScaling=self.maze_size / 2,
                physicsClientId=self.CLIENT
                )

        p.loadURDF("cube_long.urdf",
                [0, -self.maze_size / 4, wall_height / 2],
                p.getQuaternionFromEuler([0, 0, 0]),
                globalScaling=self.maze_size / 2,
                physicsClientId=self.CLIENT
                )

        p.loadURDF("cube_long.urdf",
                [-self.maze_size / 4, -self.maze_size / 2 + 1.0, wall_height / 2],
                p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                globalScaling=self.maze_size / 2 - 2.0,
                physicsClientId=self.CLIENT
                )

        p.loadURDF("cube_long.urdf",
                [self.maze_size / 4, 0, wall_height / 2],
                p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                globalScaling=self.maze_size / 2,
                physicsClientId=self.CLIENT
                )

        p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.CLIENT)

    def _computeReward(self):

        reward = 0.0
        state = self._getDroneStateVector(0)
        pos = state[0:3]


        goal_position = np.array([self.maze_size / 2 - 0.5, 0.0, pos[2]])


        distance_to_goal = np.linalg.norm(pos - goal_position)


        reward = -distance_to_goal


        if distance_to_goal < 0.2:
            reward += 100

        return reward
    
    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]

        goal_position = np.array([self.maze_size / 2 - 0.5, 0.0, pos[2]])
        if np.linalg.norm(pos - goal_position) < 0.2:
            return True


        maze_boundary = self.maze_size / 2 + 0.5
        if abs(pos[0]) > maze_boundary or abs(pos[1]) > maze_boundary:
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeTruncated(self):
   
        state = self._getDroneStateVector(0)
        pos = np.array([state[0], state[1]])

        if self.step_counter > self.PYB_FREQ * self.EPISODE_LEN_SEC:
            return True
        return False
    
    def _computeInfo(self):

        state = self._getDroneStateVector(0)
        pos = state[0:3]

        goal_position = np.array([self.maze_size / 2 - 0.5, 0.0, pos[2]])
        distance_to_goal = np.linalg.norm(pos - goal_position)

        return {"distance_to_goal": distance_to_goal}
    
    def _clipAndNormalizeState(self, state):

        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = self.maze_size / 2 + 1.0  
        MAX_Z = 1  

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
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

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

        if not np.allclose(clipped_pos_xy, state[0:2]):
            print("[WARNING] ", self.step_counter, "step，MazeAviary._clipAndNormalizeState()：clipped xy pos [{:.2f}, {:.2f}]".format(state[0], state[1]))
        if not np.isclose(clipped_pos_z, state[2]):
            print("[WARNING] ", self.step_counter, "step，MazeAviary._clipAndNormalizeState()：clipped z pos [{:.2f}]".format(state[2]))
        if not np.allclose(clipped_rp, state[7:9]):
            print("[WARNING] ", self.step_counter, "step，MazeAviary._clipAndNormalizeState()：clipped roll/pitch [{:.2f}, {:.2f}]".format(state[7], state[8]))
        if not np.allclose(clipped_vel_xy, state[10:12]):
            print("[WARNING] ", self.step_counter, "step，MazeAviary._clipAndNormalizeState()：clipped xy vel [{:.2f}, {:.2f}]".format(state[10], state[11]))
        if not np.isclose(clipped_vel_z, state[12]):
            print("[WARNING] ", self.step_counter, "step，MazeAviary._clipAndNormalizeState()：clipped z vel [{:.2f}]".format(state[12]))





