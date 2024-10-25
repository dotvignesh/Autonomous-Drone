import numpy as np
import pybullet as p
import time
from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

KEY_MAP = {
    p.B3G_UP_ARROW: [0, 0, +0.1, 0],  # Increase thrust to go up
    p.B3G_DOWN_ARROW: [0, 0, -0.1, 0],  # Decrease thrust to go down
    p.B3G_LEFT_ARROW: [0, +0.1, 0, 0],  # Increase roll to move left
    p.B3G_RIGHT_ARROW: [0, -0.1, 0, 0],  # Decrease roll to move right
    ord('z'): [0, 0, 0, +0.1],  # Increase yaw to rotate left (new key)
    ord('x'): [0, 0, 0, -0.1],  # Decrease yaw to rotate right (new key)
    ord('i'): [+0.1, 0, 0, 0],  # Increase pitch to move forward
    ord('k'): [-0.1, 0, 0, 0],  # Decrease pitch to move backward
}

def get_keyboard_input():
    """Reads keyboard input and returns a corresponding action."""
    keys = p.getKeyboardEvents()
    action = np.array([0, 0, 0, 0], dtype=np.float32)  # Initialize action as float32
    action_changed = False  # Flag to check if action has changed

    for k, v in keys.items():
        if v & p.KEY_IS_DOWN and k in KEY_MAP:
            action += KEY_MAP[k]
            action_changed = True  # Set flag if action changes
    
    action = np.clip(action, -1.0, 1.0)  # Clamping action values within [-1, 1] range
    action = action.reshape(1, 4)  # Reshape action to (1, 4)

    # Print action only if it has changed
    if action_changed:
        print("Action:", action)  # Debug print for action values

    return action

def main():
    env = HoverAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=True,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )

    print("THIS IS ACTION SPACE:", env.action_space)
    
    env.reset()

    try:
        while True:

            action = get_keyboard_input()
            

            obs, reward, done, truncated, info = env.step(action)


            if done or truncated:
                print("Episode ended. Resetting environment...")
                env.reset()

            time.sleep(0.1)  # Sleep based on the simulation time step

    except KeyboardInterrupt:
        print("Keyboard Interrupt detected. Exiting...")

    finally:
        env.close()

if __name__ == "__main__":
    main()
