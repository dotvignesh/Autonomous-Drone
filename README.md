# Autonomous Drone Navigation Using Reinforcement Learning

## Overview

This project implements and evaluates autonomous drone navigation using **Reinforcement Learning (RL)** techniques. It features the implementation of **Proximal Policy Optimization (PPO)** and **Deep Deterministic Policy Gradient (DDPG)** algorithms. The project leverages the `gym-pybullet-drones` library to simulate drone physics and custom environments, offering a realistic platform to explore drone control strategies.

### Key Features:

- **Custom Environments:** Designed with specific tasks such as hovering, obstacle navigation, gate passing, and circular path following.
- **Reinforcement Learning Algorithms:** Custom and baseline implementations of PPO and DDPG.
- **Evaluation Scenarios:** Comprehensive testing across varied tasks to analyze model performance and reward structures.
- **Hybrid Approaches:** Exploring combinations of RL with traditional PID controllers.

---

## Project Structure

### Custom Environments

Located under the `gym_pybullet_drones/envs` directory:

- **`HoverAviary.py`:** Default RL environment from `gym-pybullet-drones`.
- **`CircuitAviary.py`:** Custom environment with gates arranged in a circular path.
- **`FlyThruGateAviary.py`:** Environment with two gates and an obstacle in between.
- **`FlyThruGoalGateAviary.py`:** Environment with two gates where passing through one specific gate is the goal.

### Training Scripts

#### PPO

- **Custom Implementation:**

  - `learn_circuit.py`: Trains a PPO model to navigate a drone through a circular circuit.
  - `learn_gate.py`: Trains a PPO model to pass through gates while avoiding obstacles.
  - `learn_hover.py`: Trains a PPO model to hover at a fixed position.
  - `learn_thru_goal_gate.py`: Trains a PPO model to navigate through a specific goal gate.

- **Stable Baselines 3 (SB3):**
  - `learn_circuit_stable_baseline.py`: SB3 PPO implementation for circular circuit navigation.
  - `learn_gate_stable_baseline.py`: SB3 PPO implementation for gate passing.
  - `learn_goal_gate_sb3.py`: SB3 PPO implementation for goal-directed gate navigation.
  - `learn_hover_stable_baseline.py`: SB3 PPO implementation for hovering.

#### DDPG

- **Custom Implementation:**
  - `learn_circuit_ddpg.py`: Trains a DDPG model for circuit navigation.
  - `learn_FlyThruGate_ddpg.py`: Trains a DDPG model for navigating through gates with obstacles.
  - `learn_hover_ddpg.py`: Trains a DDPG model for stable hovering.
  - `test_circuit_ddpg.py`: Validates the trained DDPG model for circuit navigation.

---

### Testing Scripts Descriptions

#### PPO

- `test_circuit.py`: Tests PPO-trained models for circular circuit navigation.
- `test_gates.py`: Tests PPO-trained models for obstacle and gate navigation.
- `test_goal_gate.py`: Tests PPO-trained models for goal-directed gate navigation.
- `test_hover.py`: Tests PPO-trained models for hovering tasks.

#### Stable Baselines 3 (SB3)

- `test_ppo_goal_gate_sb3.py`: Tests SB3 PPO models for goal-directed gate navigation.
- `test_ppo_hover_sb3.py`: Tests SB3 PPO models for hovering tasks.
- `test_ppo_gate_sb3.py`: Tests SB3 PPO models for gate navigation with obstacles.

#### DDPG

- `learn_FlyThruGoalGateAviary_ddpg.py`: A specialized DDPG training file for goal-directed gate navigation.
- `test_FlyThruGate_ddpg.py`: Tests DDPG-trained models for navigating through gates.
- `test_hover_ddpg.py`: Tests DDPG-trained models for stable hovering.

---

## Installation (Main instructions from official repo of gym-pybullet-drone)

### Clone the Repository

```
git clone https://github.com/dotvignesh/FAI_Project.git
cd gym-pybullet-drones/
```

### Setup the Environment

1. Create and activate a Python virtual environment:
   ```
   conda create -n drones python=3.10
   conda activate drones
   ```
2. Install dependencies:
   ```
   pip3 install --upgrade pip
   pip3 install -e .
   # If needed, install build essentials:
   # sudo apt install build-essential
   ```

---

## Usage

### Training Models

Run any of the training scripts from the `Train/` folder. For example:

```
python Train/PPO/learn_circuit.py
```

### Testing Models

Use the corresponding testing scripts in the `Test/` folder. For example:

```
python Test/PPO/test_circuit.py
```

### Troubleshooting

If the training scripts cannot detect the custom environments, add the root project folder to the Python path:

```python
import sys
sys.path.append('/path/to/project-folder')
```

---

## Logs and Models

- All training logs and checkpoints are stored in the `log_dir` directory.
- Trained models are saved as `.pth` files for evaluation.

---

## Citation

```bibtex
@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control},
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      volume={},
      number={},
      pages={7512-7519},
      doi={10.1109/IROS51168.2021.9635857}
}
```

---

## Acknowledgements

This project was inspired and supported by the following resources:

- [Drone Control using Reinforcement Learning](https://github.com/phuongboi/drone-control-using-reinforcement-learning/tree/main)
- [PPO Implementation in PyTorch](https://www.youtube.com/watch?v=hlv79rcHws0&pp=ygUOcHBvIGluIHB5dG9yY2g%3D)
