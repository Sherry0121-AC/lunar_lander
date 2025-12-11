"""
LunarLander DQN with Gymnasium and TensorBoard Logging
Task 1: Reinforcement Learning Track
Author: [Your Name]
Date: 2024
"""

import gymnasium as gym
import numpy as np
import random
import os
import time
from collections import deque
from datetime import datetime

# TensorFlow 2.x with TensorBoard
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ==================== HYPERPARAMETERS ====================
TRAINING = True  # Set to True for training, False for testing

# Learning parameters
LEARNING_RATE = 0.0005
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Training parameters
LEARNING_EPISODES = 1000 if TRAINING else 0  # 改为1000以便快速测试
TESTING_EPISODES = 100 if not TRAINING else 0
REPLAY_BUFFER_SIZE = 100000
REPLAY_BUFFER_BATCH_SIZE = 64
MINIMUM_REWARD = -250

# Environment parameters
STATE_SIZE = 8
NUMBER_OF_ACTIONS = 4

# File paths
WEIGHTS_FILENAME = './weights/lunar_lander_dqn.h5'
LOG_DIR = './logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")

# ==================== DQN AGENT ====================
class DQNAgent:
    def __init__(self, training=True):
        self.training = training
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        
        # Exploration parameters
        self.epsilon = EPSILON_START if training else 0.0
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_END
        
        # Create Q-network and target network
        self.model = self._build_q_network()
        self.target_model = self._build_q_network()
        self.target_model.set_weights(self.model.get_weights())
        
        # Optimizer and loss function
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.loss_fn = keras.losses.Huber()
        
        # For TensorBoard logging - 使用TensorFlow的SummaryWriter
        if training:
            # 确保日志目录存在
            os.makedirs(LOG_DIR, exist_ok=True)
            self.writer = tf.summary.create_file_writer(LOG_DIR)
            print(f"📁 TensorBoard日志目录已创建: {LOG_DIR}")
        else:
            self.writer = None
            
        self.episode_rewards = []
        self.episode_losses = []
        
        # Load weights if testing
        if not training:
            self.load_weights(WEIGHTS_FILENAME)
    
    def _build_q_network(self):
        """Build Deep Q-Network using Keras"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(STATE_SIZE,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(NUMBER_OF_ACTIONS, activation='linear')
        ])
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        if self.training:
            self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if not self.training or np.random.random() > self.epsilon:
            state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
            q_values = self.model(state_tensor, training=False)
            return np.argmax(q_values[0].numpy())
        else:
            return np.random.randint(NUMBER_OF_ACTIONS)
    
    def train(self):
        """Train on a batch from replay buffer"""
        if len(self.memory) < REPLAY_BUFFER_BATCH_SIZE:
            return 0
        
        # Sample batch from memory
        batch = random.sample(self.memory, REPLAY_BUFFER_BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Compute target Q-values
        next_q_values = self.target_model(next_states, training=False)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        target_q = rewards + (1 - dones) * DISCOUNT_FACTOR * max_next_q
        
        # Compute current Q-values
        with tf.GradientTape() as tape:
            all_q_values = self.model(states, training=True)
            q_values = tf.reduce_sum(
                all_q_values * tf.one_hot(actions, NUMBER_OF_ACTIONS), 
                axis=1
            )
            loss = self.loss_fn(target_q, q_values)
        
        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss.numpy()
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def update_epsilon(self):
        """Decay exploration rate"""
        if self.training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_weights(self, filename):
        """Save model weights"""
        self.model.save_weights(filename)
        print(f"✅ Model saved to {filename}")
    
    def load_weights(self, filename):
        """Load model weights"""
        if os.path.exists(filename):
            self.model.load_weights(filename)
            self.target_model.set_weights(self.model.get_weights())
            print(f"✅ Model loaded from {filename}")
        else:
            print(f"⚠️  No weights found at {filename}, using random initialization")
    
    def log_metrics(self, episode, reward, loss, steps, epsilon):
        """Log metrics to TensorBoard - 使用TensorFlow的API"""
        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar('Reward/Episode', reward, step=episode)
                tf.summary.scalar('Loss/Episode', loss, step=episode)
                tf.summary.scalar('Steps/Episode', steps, step=episode)
                tf.summary.scalar('Epsilon', epsilon, step=episode)
                
                # Log average rewards every 10 episodes
                if len(self.episode_rewards) >= 10:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    tf.summary.scalar('Reward/Average_10', avg_reward, step=episode)
                
                if len(self.episode_losses) >= 10:
                    avg_loss = np.mean(self.episode_losses[-10:])
                    tf.summary.scalar('Loss/Average_10', avg_loss, step=episode)
                
                # 刷新写入器以确保数据被保存
                self.writer.flush()
    
    def close_writer(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()
            print("📊 TensorBoard writer closed")

# ==================== TRAINING FUNCTION ====================
def train_agent():
    """Main training function"""
    print("=" * 60)
    print("🚀 Starting LunarLander DQN Training")
    print("=" * 60)
    print(f"📊 Hyperparameters:")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Discount Factor: {DISCOUNT_FACTOR}")
    print(f"   Epsilon Decay: {EPSILON_DECAY}")
    print(f"   Batch Size: {REPLAY_BUFFER_BATCH_SIZE}")
    print(f"   Max Episodes: {LEARNING_EPISODES}")
    print(f"   Log Directory: {LOG_DIR}")
    print("=" * 60)
    
    # 检查日志目录
    print(f"📁 检查日志目录: {LOG_DIR}")
    print(f"📁 目录是否存在: {os.path.exists(LOG_DIR)}")
    
    # Create environment with Gymnasium
    env = gym.make(
        "LunarLander-v2",
        render_mode=None  # No rendering during training for speed
    )
    
    # Initialize agent
    agent = DQNAgent(training=True)
    
    # Track metrics
    all_rewards = []
    episode_lengths = []
    average_rewards = deque(maxlen=100)
    
    # Create weights directory if it doesn't exist
    os.makedirs('./weights', exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # Training loop
    for episode in range(LEARNING_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        for t in range(1000):  # Max steps per episode
            # Select and execute action
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train on batch
            loss = agent.train()
            if loss > 0:
                episode_loss += loss
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done or episode_reward < MINIMUM_REWARD:
                break
        
        # Update target network and epsilon
        agent.update_target_network()
        agent.update_epsilon()
        
        # Calculate average loss per step
        avg_loss_per_step = episode_loss / max(steps, 1)
        
        # Store metrics
        all_rewards.append(episode_reward)
        episode_lengths.append(steps)
        average_rewards.append(episode_reward)
        agent.episode_rewards.append(episode_reward)
        agent.episode_losses.append(avg_loss_per_step)
        
        # Log metrics to TensorBoard
        agent.log_metrics(
            episode, 
            episode_reward, 
            avg_loss_per_step, 
            steps, 
            agent.epsilon
        )
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(average_rewards) if average_rewards else 0
            elapsed_time = time.time() - start_time
            episodes_per_sec = (episode + 1) / elapsed_time if elapsed_time > 0 else 0
            
            print(f"📈 Episode {episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Steps: {steps:4d} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Memory: {len(agent.memory):6d} | "
                  f"EPS: {episodes_per_sec:.2f}/s")
        
        # Save model every 100 episodes
        if episode % 100 == 0 and episode > 0:
            checkpoint_file = f"./weights/checkpoint_ep{episode}.h5"
            agent.save_weights(checkpoint_file)
            
            # 检查TensorBoard事件文件
            event_files = [f for f in os.listdir(LOG_DIR) if 'tfevents' in f]
            print(f"📊 TensorBoard事件文件: {len(event_files)} 个文件")
        
        # Early stopping if performance is good
        if len(average_rewards) == 100 and np.mean(average_rewards) > 200:
            print(f"🎉 Early stopping at episode {episode}: Average reward > 200!")
            break
    
    # Save final model
    agent.save_weights(WEIGHTS_FILENAME)
    
    # Close TensorBoard writer
    agent.close_writer()
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"⏱️  Total training time: {total_time:.2f} seconds")
    
    env.close()
    
    # Plot training results
    plot_training_results(all_rewards, episode_lengths)
    
    # 显示TensorBoard使用说明
    print("\n" + "=" * 60)
    print("📊 TENSORBOARD INSTRUCTIONS:")
    print("=" * 60)
    print("1. 在新终端中运行:")
    print("   tensorboard --logdir logs/")
    print("2. 在浏览器中打开:")
    print("   http://localhost:6006")
    print("3. 查看以下指标:")
    print("   - Reward/Episode: 每个回合的奖励")
    print("   - Loss/Episode: 每个回合的损失")
    print("   - Steps/Episode: 每个回合的步数")
    print("   - Epsilon: 探索率衰减")
    print("   - Reward/Average_10: 最近10回合平均奖励")
    print("=" * 60)
    
    return all_rewards

# ==================== TESTING FUNCTION ====================
def test_agent():
    """Test the trained agent"""
    print("=" * 60)
    print("🧪 Testing LunarLander DQN Agent")
    print("=" * 60)
    
    # 检查模型文件是否存在
    if not os.path.exists(WEIGHTS_FILENAME):
        print(f"❌ 模型文件不存在: {WEIGHTS_FILENAME}")
        print("请先运行训练模式 (TRAINING = True)")
        return []
    
    # Create environment with rendering
    env = gym.make(
        "LunarLander-v2",
        render_mode="human"  # Enable rendering for visualization
    )
    
    # Initialize agent
    agent = DQNAgent(training=False)
    
    # Track metrics
    test_rewards = []
    
    for episode in range(TESTING_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for t in range(1000):
            # Select action (no exploration during testing)
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done or episode_reward < MINIMUM_REWARD:
                break
        
        test_rewards.append(episode_reward)
        print(f"🧪 Test Episode {episode:3d} | Reward: {episode_reward:7.2f} | Steps: {steps:4d}")
    
    env.close()
    
    # Print test statistics
    print("=" * 60)
    print("📊 Test Results:")
    print(f"   Total Episodes: {len(test_rewards)}")
    print(f"   Average Reward: {np.mean(test_rewards):.2f}")
    print(f"   Std Deviation: {np.std(test_rewards):.2f}")
    print(f"   Minimum Reward: {np.min(test_rewards):.2f}")
    print(f"   Maximum Reward: {np.max(test_rewards):.2f}")
    
    # Calculate success rate
    success_count = sum(1 for r in test_rewards if r > 200)
    success_rate = (success_count / len(test_rewards)) * 100 if test_rewards else 0
    print(f"   Success Rate (>200): {success_rate:.1f}% ({success_count}/{len(test_rewards)})")
    print("=" * 60)
    
    return test_rewards

# ==================== VISUALIZATION ====================
def plot_training_results(rewards, steps):
    """Plot training results"""
    if not rewards:
        print("⚠️  No training data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.6, linewidth=1)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Rewards per Episode', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add moving average
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', linewidth=2, 
                label=f'{window_size}-episode moving average')
        ax1.legend(fontsize=10)
    
    # Add horizontal line for success threshold
    ax1.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Success threshold (200)')
    ax1.legend(fontsize=10)
    
    # Plot steps
    ax2.plot(steps, alpha=0.6, color='green', linewidth=1)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Steps', fontsize=12)
    ax2.set_title('Episode Length (Steps)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'./training_results_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📈 Training plot saved to {filename}")
    
    # Show the plot (如果运行在GUI环境)
    try:
        plt.show()
    except:
        print("📊 Plot generated but not displayed (running in non-GUI environment)")

# ==================== VERIFY TENSORBOARD ====================
def verify_tensorboard_setup():
    """Verify TensorBoard setup"""
    print("🔍 Verifying TensorBoard setup...")
    
    # 检查日志目录
    if not os.path.exists('./logs'):
        os.makedirs('./logs', exist_ok=True)
        print("✅ Created logs directory")
    
    # 创建测试事件文件
    test_log_dir = './logs/test_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(test_log_dir, exist_ok=True)
    
    writer = tf.summary.create_file_writer(test_log_dir)
    
    # 写入测试数据
    with writer.as_default():
        for i in range(10):
            tf.summary.scalar('test_scalar', i * 1.5, step=i)
        writer.flush()
    
    writer.close()
    
    # 检查是否创建了事件文件
    import glob
    event_files = glob.glob(f"{test_log_dir}/*.tfevents*")
    
    if event_files:
        print(f"✅ TensorBoard setup verified: {len(event_files)} event file(s) created")
        print(f"📁 Test log directory: {test_log_dir}")
    else:
        print("❌ TensorBoard setup failed: No event files created")
    
    return len(event_files) > 0

# ==================== MAIN ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🌙 LunarLander-v2 DQN Implementation")
    print("📚 Task 1: Reinforcement Learning Track")
    print("=" * 60)
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # 创建必要的目录
    os.makedirs('./weights', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # 验证TensorBoard设置
    tensorboard_ok = verify_tensorboard_setup()
    if not tensorboard_ok:
        print("⚠️  TensorBoard setup may have issues, but continuing anyway...")
    
    print("\n" + "=" * 60)
    
    if TRAINING:
        # 训练模式
        print("🎯 MODE: TRAINING")
        print("=" * 60)
        rewards = train_agent()
        
        # 训练后快速测试
        if rewards and len(rewards) > 0:
            print("\n" + "=" * 60)
            print("🔍 Quick Test after Training")
            print("=" * 60)
            
            # 临时切换到测试模式
            original_training = TRAINING
            TRAINING = False
            
            try:
                test_agent()
            except Exception as e:
                print(f"⚠️  Quick test failed: {e}")
            
            TRAINING = original_training
    else:
        # 测试模式
        print("🎯 MODE: TESTING")
        print("=" * 60)
        test_rewards = test_agent()
    
    print("\n" + "=" * 60)
    print("🎉 Program completed successfully!")
    print("=" * 60)