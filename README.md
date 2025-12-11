# LunarLander DQN Project

## 📋 Project Overview
A Deep Q-Network (DQN) agent trained to solve the LunarLander-v2 environment from Gymnasium. The trained model achieves an average reward of 246.23 with a 90% success rate.

## 🛠️ How to Run

### 1. Setup Environment
```bash
git clone https://github.com/Sherry0121-AC/lunar_lander.git
cd lunar_lander

pip install -r requirements.txt
```

### 2. Test the Pre-trained Model 
Run the test script to see the trained agent in action:
```bash
python test_trained_model.py
```
Then select an option:
1: Quick Test (5 episodes with rendering)

2: Full Test (10 episodes with rendering)

3: Record Demo (generate a GIF)

4: No-render Test (for server environments)

### 3. Train from Scratch
If you want to train the model yourself:
```bash
python lunar_lander.py
```

### 4. Visualize Training
```bash
tensorboard --logdir logs/
```
