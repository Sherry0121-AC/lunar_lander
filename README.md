# LunarLander DQN with Gymnasium

## 📋 Project Overview
This project implements a Deep Q-Network (DQN) agent for the LunarLander-v2 environment using Gymnasium API.

## 🚀 Features
- Modern implementation using Gymnasium (not deprecated Gym)
- Deep Q-Network with experience replay and target network
- TensorBoard logging for training visualization
- Comprehensive hyperparameter tracking
- Training and testing modes

## 📊 Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.0005 | Controls how much to update weights |
| Discount Factor (γ) | 0.99 | Importance of future rewards |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon Decay | 0.995 | Exploration decay rate |
| Batch Size | 64 | Training batch size |
| Replay Buffer | 100,000 | Experience replay memory size |

## 📈 How to Use

### 1. Install dependencies
```bash
pip install -r requirements.txt