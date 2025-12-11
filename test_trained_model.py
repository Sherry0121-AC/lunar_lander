# test_trained_model.py
"""
独立测试训练好的LunarLander DQN模型
不需要重新训练
"""

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def load_trained_model():
    """加载训练好的模型"""
    model_path = './weights/lunar_lander_dqn.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    # 构建相同的网络结构
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(8,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(4, activation='linear')
    ])
    
    # 加载权重
    model.load_weights(model_path)
    print(f"✅ 模型加载成功: {model_path}")
    return model

def test_model(num_episodes=10, render=True):
    """测试训练好的模型"""
    print("=" * 60)
    print("🧪 Testing Trained LunarLander DQN Model")
    print("=" * 60)
    
    # 加载模型
    model = load_trained_model()
    if model is None:
        return
    
    # 创建环境
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    
    test_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for t in range(1000):
            # 选择最优动作（无探索）
            state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
            q_values = model(state_tensor, training=False)
            action = np.argmax(q_values[0].numpy())
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        print(f"🧪 Test Episode {episode:3d} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Steps: {steps:4d}")
    
    env.close()
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    
    if test_rewards:
        avg_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        max_reward = np.max(test_rewards)
        min_reward = np.min(test_rewards)
        
        # 计算成功率（>200为成功着陆）
        success_count = sum(1 for r in test_rewards if r > 200)
        success_rate = (success_count / len(test_rewards)) * 100
        
        print(f"📈 测试回合数: {len(test_rewards)}")
        print(f"💰 平均奖励: {avg_reward:.2f}")
        print(f"📊 标准差: {std_reward:.2f}")
        print(f"🏆 最高奖励: {max_reward:.2f}")
        print(f"📉 最低奖励: {min_reward:.2f}")
        print(f"✅ 成功率 (>200): {success_rate:.1f}% ({success_count}/{len(test_rewards)})")
        
        # 性能评估
        if avg_reward > 200:
            print("🎉 性能评级: 优秀 - 智能体已学会稳定着陆!")
        elif avg_reward > 100:
            print("👍 性能评级: 良好 - 智能体基本掌握着陆")
        elif avg_reward > 0:
            print("👌 性能评级: 及格 - 智能体开始学习")
        else:
            print("⚠️  性能评级: 需改进 - 智能体仍需训练")
    
    print("=" * 60)
    return test_rewards

def record_demo():
    """录制一个演示回合"""
    print("🎥 Recording demonstration episode...")
    
    model = load_trained_model()
    if model is None:
        return
    
    # 创建环境
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    state, _ = env.reset()
    
    frames = []
    episode_reward = 0
    
    for t in range(1000):
        # 选择动作
        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        q_values = model(state_tensor, training=False)
        action = np.argmax(q_values[0].numpy())
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 记录帧（用于制作GIF）
        frame = env.render()
        frames.append(frame)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    env.close()
    
    print(f"🎬 演示回合录制完成!")
    print(f"💰 演示奖励: {episode_reward:.2f}")
    print(f"📷 录制帧数: {len(frames)}")
    
    # 可以保存为GIF（需要安装imageio）
    try:
        import imageio
        imageio.mimsave('./lunar_lander_demo.gif', frames, fps=30)
        print("✅ 演示已保存为: ./lunar_lander_demo.gif")
    except:
        print("⚠️  无法保存GIF，请安装: pip install imageio")

if __name__ == "__main__":
    print("🌙 LunarLander-v2 DQN Model Tester")
    print("📚 Testing pre-trained model without re-training")
    print("=" * 60)
    
    # 选择测试模式
    print("选择测试模式:")
    print("1. 快速测试 (5回合)")
    print("2. 完整测试 (10回合)")
    print("3. 录制演示")
    print("4. 无渲染测试")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    if choice == "1":
        test_model(num_episodes=5, render=True)
    elif choice == "2":
        test_model(num_episodes=10, render=True)
    elif choice == "3":
        record_demo()
    elif choice == "4":
        test_model(num_episodes=10, render=False)
    else:
        print("使用默认设置: 快速测试")
        test_model(num_episodes=5, render=True)