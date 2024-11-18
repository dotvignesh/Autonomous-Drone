## Features
- **Custom Environment**: DynamicObstacleHoverAviary simulates a drone navigating around moving obstacles.
- **Reinforcement Learning**: PPO algorithm trains the drone for optimal navigation.
- **GPU Support**: AMD GPU acceleration enabled via DirectML for faster training.

## File Structure
- **DynamicObstacleHoverAviary.py**: Custom environment with dynamic obstacles.
- **train_ppo.py**: Script to train the PPO model.
- **test_trained_model.py**: Script to evaluate the trained model.
- **models/**: Directory for saving trained models.
## Training
python train_ppo.py
##  Evaluation
python eval_ppo.py
## Requirements
- **Python 3.10**
- **Gymnasium**
- **Stable-Baselines3**
- **PyBullet**
- **torch-directml** (for AMD GPU support) if you have nvadia, you can just use torch directly 
