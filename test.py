import numpy as np
import pybullet as p
import time
from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

KEY_MAP = {
    p.B3G_UP_ARROW: [0, 0, +0.05, 0],  
    p.B3G_DOWN_ARROW: [0, 0, -0.05, 0],
    p.B3G_LEFT_ARROW: [0, +0.05, 0, 0],  
    p.B3G_RIGHT_ARROW: [0, -0.05, 0, 0],
    ord('z'): [0, 0, 0, +0.05],  
    ord('x'): [0, 0, 0, -0.05],  
    ord('i'): [+0.05, 0, 0, 0],  
    ord('k'): [-0.05, 0, 0, 0],  
}

def get_keyboard_input(current_action, client_id):
    """Reads keyboard input and returns a smoothly transitioning action."""
    if p.getConnectionInfo(physicsClientId=client_id)['isConnected']:
        keys = p.getKeyboardEvents()
        action = np.array([0, 0, 0, 0], dtype=np.float32)

        for k, v in keys.items():
            if v & p.KEY_IS_DOWN and k in KEY_MAP:
                action += KEY_MAP[k]

        # Blend with the current action for smooth transition
        new_action = 0.8 * current_action + 0.2 * action  
        new_action = np.clip(new_action, -1.0, 1.0)  
        return new_action.reshape(1, 4)
    else:
        raise ConnectionError("Not connected to the physics server.")

def create_obstacles(client_id):
    """Creates a more detailed environment with buildings and walls."""
    # Define parameters for the buildings and walls
    building_heights = [1.5, 2.0, 3.0]  # Varied building heights
    building_widths = [0.5, 0.6]  # Smaller widths for narrow pathways
    wall_height = 0.75  # Shorter walls to create separation without blocking visibility
    building_colors = [[0.7, 0.2, 0.2, 1], [0.2, 0.7, 0.2, 1], [0.2, 0.2, 0.7, 1]]  # Varied colors

    # Define building and wall positions to form a navigable pattern
    building_positions = [
        [-2.5, -2.5, building_heights[0] / 2], [-1.0, -2.5, building_heights[1] / 2],
        [1.0, -2.5, building_heights[2] / 2], [2.5, -2.5, building_heights[0] / 2],
        [-2.5, 0, building_heights[1] / 2], [2.5, 0, building_heights[2] / 2],
        [-2.5, 2.5, building_heights[2] / 2], [0, 2.5, building_heights[0] / 2], [2.5, 2.5, building_heights[1] / 2]
    ]
    wall_positions = [
        [-1.5, 1.5, wall_height / 2], [1.5, -1.5, wall_height / 2],
        [0, -0.75, wall_height / 2], [0, 0.75, wall_height / 2],
        [-0.75, 0, wall_height / 2], [0.75, 0, wall_height / 2]
    ]

    # Create buildings
    for i, pos in enumerate(building_positions):
        height = building_heights[i % len(building_heights)]
        width = building_widths[i % len(building_widths)]
        
        building_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[width, width, height / 2])
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[width, width, height / 2],
            rgbaColor=building_colors[i % len(building_colors)]
        )
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=building_shape, baseVisualShapeIndex=visual_shape,
            basePosition=[pos[0], pos[1], height / 2]
        )

    # Create narrow, lower walls to create pathways
    for pos in wall_positions:
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, wall_height / 2])
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.5, wall_height / 2],
            rgbaColor=[0.6, 0.6, 0.6, 1]  # Neutral gray color for walls
        )
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=wall_shape, baseVisualShapeIndex=visual_shape, basePosition=pos
        )
    pass

def check_collision(client_id, drone_id):
    """Checks for collisions between the drone and obstacles."""
    collision = False
    if p.getConnectionInfo(physicsClientId=client_id)['isConnected']:
        for i in range(p.getNumBodies(physicsClientId=client_id)):
            if i != drone_id:
                contacts = p.getContactPoints(bodyA=drone_id, bodyB=i, physicsClientId=client_id)
                if contacts:
                    collision = True
                    break
    return collision

def main():
    try:
        env = HoverAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            gui=True,
            obs=ObservationType.KIN,
            act=ActionType.RPM
        )
        
        client_id = env.CLIENT
        drone_id = env.DRONE_IDS[0]

        if not p.getConnectionInfo(physicsClientId=client_id)['isConnected']:
            raise ConnectionError("Failed to connect to the physics server.")

        env.reset()
        create_obstacles(client_id)

        current_action = np.array([0, 0, 0, 0], dtype=np.float32)

        while True:
            try:
                # Smooth keyboard input for smoother action
                current_action = get_keyboard_input(current_action, client_id)
                obs, reward, done, truncated, info = env.step(current_action)

                if check_collision(client_id, drone_id):
                    print("Collision detected! Ending episode.")
                    done = True

                if done or truncated:
                    print("Episode ended. Resetting environment...")
                    time.sleep(0.5)  # Small delay after reset to stabilize
                    env.reset()
                    create_obstacles(client_id)

                time.sleep(0.1)  # Time step to control action speed

            except ConnectionError:
                print("Lost connection to physics server.")
                break

    except KeyboardInterrupt:
        print("Keyboard Interrupt detected. Exiting...")

    finally:
        if hasattr(env, 'CLIENT') and p.getConnectionInfo(physicsClientId=env.CLIENT)['isConnected']:
            env.close()
        else:
            print("Environment already disconnected from the physics server.")

if __name__ == "__main__":
    main()





