import numpy as np
import gym
import gym_pybullet_drones
from gym_pybullet_drones.envs.CircuitAviary import CircuitAviary
from gym_pybullet_drones.utils.enums import ActionType

class SimpleModel:
    """A simple model that sends a list of target [x, y, z] coordinates to the environment."""
    
    def __init__(self, gates_positions, circuit_radius):
        # Calculate the middle points between gates considering the radius
        self.target_positions = []
        self.circuit_radius = circuit_radius
        self.calculate_middle_points(gates_positions)
        self.current_target_idx = 0  # Start with the first target position

    def calculate_middle_points(self, gates_positions):
        """Calculate the middle points between consecutive gates while accounting for the circuit radius."""
        for i in range(len(gates_positions)):
            # Get the current and next gate (wrap around to the first one after the last gate)
            current_gate = gates_positions[i]
            next_gate = gates_positions[(i + 1) % len(gates_positions)]
            
            # Calculate the midpoint between the two gates
            midpoint = [(current_gate[0] + next_gate[0]) / 2,
                        (current_gate[1] + next_gate[1]) / 2,
                        0.55]  # Keeping the height constant for simplicity
            
            # Ensure the midpoint lies on the circle by adjusting the x and y to match the circuit radius
            dist = np.sqrt(midpoint[0]**2 + midpoint[1]**2)
            midpoint[0] = (midpoint[0] / dist) * self.circuit_radius
            midpoint[1] = (midpoint[1] / dist) * self.circuit_radius
            
            self.target_positions.append(midpoint)

    def get_action(self, state):
        """Cycle through the list of target positions."""
        # Return the current target position
        action = self.target_positions[self.current_target_idx]
        
        # Move to the next target position in the list
        self.current_target_idx = (self.current_target_idx + 1) % len(self.target_positions)
        
        return action


# Custom environment setup using ActionType.PID
env = CircuitAviary(
    act=ActionType.PID,  # Using PID controller for action type
    gui=True,
    record=False
)

# Gate positions from the environment setup
gates_positions = []
NUM_GATES = 8  # Number of gates
CIRCUIT_RADIUS = 2.0  # Radius of the circuit

# Generate gate positions along circular path
for i in range(NUM_GATES):
    angle = (2 * np.pi * i) / NUM_GATES
    x = CIRCUIT_RADIUS * np.cos(angle)
    x += 0.3 * np.cos(angle)
    y = CIRCUIT_RADIUS * np.sin(angle)
    y += y + 0.3 * np.sin(angle)
    
    # Get midpoint for each gate
    # midpoint_x, midpoint_y = get_pillar_midpoint(x, y, angle, h=1)
    gates_positions.append([x, y])

# Get the circuit radius
circuit_radius = CIRCUIT_RADIUS

# Initialize the model with the list of target positions (midpoints between gates)
model = SimpleModel(gates_positions=gates_positions, circuit_radius=circuit_radius)

# Reset the environment
state = env.reset()

# Run the environment with the model controlling the drone
done = False
while not done:
    # Get the model's action (next target position)
    action = model.get_action(state)
    
    # Take a step in the environment with the target [x, y, z]
    state, reward, done, info, _ = env.step(action)
    
    # Optionally print state or reward for debugging purposes
    print(f"State: {state}, Reward: {reward}")

# Close the environment after the episode is done
env.close()