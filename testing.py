import os
import time
from datetime import datetime
import numpy as np
from gym_pybullet_drones.envs.CircuitAviary import CircuitAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Constants for default settings
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

def setup_environment(output_folder):
    """Create the Circuit Aviary environment and ensure output folder exists."""
    output_dir = os.path.join(output_folder, f'recording_{datetime.now().strftime("%m.%d.%Y_%H.%M.%S")}')
    os.makedirs(output_dir, exist_ok=True)
    return CircuitAviary(gui=DEFAULT_GUI, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=DEFAULT_RECORD_VIDEO)

def horizontal_circle_motion(t, base_rpm=4100, amplitude=200):
    """Generate RPMs for horizontal circular motion."""
    # All rotors maintain similar base RPM for stable height
    rpms = np.array([base_rpm, base_rpm, base_rpm, base_rpm])
    
    # Add differential RPMs for horizontal rotation
    # Front left and back right vs front right and back left
    differential = amplitude * np.sin(t)
    rpms[[0, 3]] += differential  # Front left and back right
    rpms[[1, 2]] -= differential  # Front right and back left
    
    return np.expand_dims(rpms, axis=0)

def main():
    """Main function to demonstrate horizontal circular flight."""
    print("===== Starting Horizontal Circular Flight Demo =====")

    # Initialize environment
    env = setup_environment(DEFAULT_OUTPUT_FOLDER)
    
    # Initial reset
    obs, _ = env.reset(seed=42, options={})
    
    # Flight parameters
    duration = 5  # seconds
    steps = int(duration * env.CTRL_FREQ)
    frame_start = time.time()
    
    # Give some time to stabilize height first
    print("Stabilizing height...")
    for _ in range(int(env.CTRL_FREQ)):  # 1 second stabilization
        action = np.array([[4100, 4100, 4100, 4100]])  # Stable hover
        obs, _, _, _, _ = env.step(action)
        env.render()
        
    print("Starting circular motion...")
    # Main control loop for circular motion
    for step in range(steps):
        # Generate horizontal circular motion
        t = 2 * np.pi * step / steps
        action = horizontal_circle_motion(t)
            
        # Apply action to environment
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Render and sync
        env.render()
        sync(step, frame_start, env.CTRL_TIMESTEP)
        
        if terminated or truncated:
            break

    print("===== Flight Demo Complete =====")
    env.close()

if __name__ == '__main__':
    main()