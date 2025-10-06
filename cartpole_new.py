# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import random
# from collections import deque, namedtuple
# import os

# # Set random seeds for reproducibility
# np.random.seed(42)
# torch.manual_seed(42)
# random.seed(42)

# # Check if CUDA is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Experience tuple
# Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# class DQNNetwork(nn.Module):
#     """Deep Q-Network"""
#     def __init__(self, state_size, action_size, hidden_size=128):
#         super(DQNNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, 64)
#         self.fc4 = nn.Linear(64, action_size)
#         self.dropout = nn.Dropout(0.2)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

# class ReplayBuffer:
#     """Experience Replay Buffer"""
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, done):
#         experience = Experience(state, action, reward, next_state, done)
#         self.buffer.append(experience)
    
#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)
    
#     def __len__(self):
#         return len(self.buffer)

# class DQNAgent:
#     def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
#                  epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, 
#                  batch_size=64, target_update_frequency=100):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.batch_size = batch_size
#         self.target_update_frequency = target_update_frequency
#         self.train_start = 1000
        
#         # Neural networks
#         self.q_network = DQNNetwork(state_size, action_size).to(device)
#         self.target_network = DQNNetwork(state_size, action_size).to(device)
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
#         # Replay buffer
#         self.memory = ReplayBuffer(buffer_size)
        
#         # Initialize target network
#         self.update_target_network()
        
#         # Training metrics
#         self.loss_history = []
#         self.step_count = 0
    
#     def update_target_network(self):
#         """Copy weights from main network to target network"""
#         self.target_network.load_state_dict(self.q_network.state_dict())
    
#     def remember(self, state, action, reward, next_state, done):
#         """Store experience in replay buffer"""
#         self.memory.push(state, action, reward, next_state, done)
    
#     def act(self, state):
#         """Choose action using epsilon-greedy policy"""
#         if random.random() <= self.epsilon:
#             return random.randrange(self.action_size)
        
#         with torch.no_grad():
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#             q_values = self.q_network(state_tensor)
#             return q_values.max(1)[1].item()
    
#     def replay(self):
#         """Train the model on a batch of experiences"""
#         if len(self.memory) < self.train_start:
#             return
        
#         # Sample batch
#         experiences = self.memory.sample(self.batch_size)
        
#         # Convert to tensors
#         states = torch.FloatTensor([e.state for e in experiences]).to(device)
#         actions = torch.LongTensor([e.action for e in experiences]).to(device)
#         rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
#         next_states = torch.FloatTensor([e.next_state for e in experiences]).to(device)
#         dones = torch.BoolTensor([e.done for e in experiences]).to(device)
        
#         # Current Q values
#         current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
#         # Next Q values from target network
#         with torch.no_grad():
#             next_q_values = self.target_network(next_states).max(1)[0]
#             target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
#         # Compute loss
#         loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
#         # Optimize
#         self.optimizer.zero_grad()
#         loss.backward()
#         # Gradient clipping for stability
#         torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
#         self.optimizer.step()
        
#         # Store loss
#         self.loss_history.append(loss.item())
        
#         # Update target network
#         self.step_count += 1
#         if self.step_count % self.target_update_frequency == 0:
#             self.update_target_network()
        
#         # Decay epsilon
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
    
#     def save(self, filename):
#         """Save the model"""
#         torch.save({
#             'q_network_state_dict': self.q_network.state_dict(),
#             'target_network_state_dict': self.target_network.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'epsilon': self.epsilon,
#             'step_count': self.step_count
#         }, filename)
#         print(f"Model saved to {filename}")
    
#     def load(self, filename):
#         """Load the model"""
#         checkpoint = torch.load(filename, map_location=device)
#         self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
#         self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.epsilon = checkpoint['epsilon']
#         self.step_count = checkpoint['step_count']
#         print(f"Model loaded from {filename}")

# def plot_training_results(episode_rewards, avg_rewards, loss_history=None, save_path='training_results.png'):
#     """Plot comprehensive training results"""
#     if loss_history and len(loss_history) > 0:
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
#     else:
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
#     # Plot episode rewards
#     ax1.plot(episode_rewards, alpha=0.6, color='blue', linewidth=1, label='Episode Reward')
#     ax1.set_title('Episode Rewards', fontsize=14, fontweight='bold')
#     ax1.set_xlabel('Episode')
#     ax1.set_ylabel('Reward')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
    
#     # Plot average rewards
#     ax2.plot(avg_rewards, color='red', linewidth=2, label='Avg Reward (100 episodes)')
#     ax2.axhline(y=195, color='green', linestyle='--', linewidth=2, label='Solved (195)')
#     ax2.set_title('Average Rewards', fontsize=14, fontweight='bold')
#     ax2.set_xlabel('Episode')
#     ax2.set_ylabel('Average Reward')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     if loss_history and len(loss_history) > 0:
#         # Plot training loss
#         ax3.plot(loss_history, color='orange', alpha=0.7, linewidth=1)
#         ax3.set_title('Training Loss', fontsize=14, fontweight='bold')
#         ax3.set_xlabel('Training Step')
#         ax3.set_ylabel('MSE Loss')
#         ax3.grid(True, alpha=0.3)
        
#         # Plot smoothed loss
#         window_size = min(100, len(loss_history) // 10)
#         if window_size > 1:
#             smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
#             ax4.plot(smoothed_loss, color='purple', linewidth=2)
#             ax4.set_title(f'Smoothed Loss (window={window_size})', fontsize=14, fontweight='bold')
#             ax4.set_xlabel('Training Step')
#             ax4.set_ylabel('Smoothed MSE Loss')
#             ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()

# def train_dqn_agent(episodes=1000, render_frequency=100, save_frequency=200):
#     """Train the DQN agent"""
#     # Create environment
#     env = gym.make('CartPole-v1', render_mode='human')
    
#     # Get environment dimensions
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
    
#     print(f"State size: {state_size}")
#     print(f"Action size: {action_size}")
    
#     # Create agent
#     agent = DQNAgent(state_size, action_size)
    
#     # Training metrics
#     episode_rewards = []
#     avg_rewards = []
#     solved = False
    
#     print("Starting training...")
#     print("=" * 50)
    
#     for episode in range(episodes):
#         state, _ = env.reset()
#         total_reward = 0
#         steps = 0
        
#         while True:
#             # Choose action
#             action = agent.act(state)
            
#             # Take action
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
            
#             # Store experience
#             agent.remember(state, action, reward, next_state, done)
            
#             # Train agent
#             agent.replay()
            
#             state = next_state
#             total_reward += reward
#             steps += 1
            
#             if done:
#                 break
        
#         # Record metrics
#         episode_rewards.append(total_reward)
        
#         # Calculate average reward over last 100 episodes
#         if len(episode_rewards) >= 100:
#             avg_reward = np.mean(episode_rewards[-100:])
#             avg_rewards.append(avg_reward)
            
#             # Check if solved
#             if avg_reward >= 195 and not solved:
#                 print(f"\n🎉 Environment solved in {episode} episodes!")
#                 print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
#                 solved = True
#                 agent.save(f'dqn_cartpole_solved_episode_{episode}.pth')
#         else:
#             avg_rewards.append(np.mean(episode_rewards))
        
#         # Print progress
#         if episode % 50 == 0:
#             avg_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
#             print(f"Episode {episode:4d} | Reward: {total_reward:3.0f} | "
#                   f"Avg(100): {avg_100:6.2f} | Epsilon: {agent.epsilon:.3f} | "
#                   f"Memory: {len(agent.memory):5d}")
        
#         # Render periodically
#         if episode % render_frequency == 0 and episode > 0:
#             print(f"Rendering episode {episode}...")
        
#         # Save model periodically
#         if episode % save_frequency == 0 and episode > 0:
#             agent.save(f'dqn_cartpole_episode_{episode}.pth')
    
#     env.close()
    
#     # Save final model
#     agent.save('dqn_cartpole_final.pth')
    
#     # Plot results
#     plot_training_results(episode_rewards, avg_rewards, agent.loss_history)
    
#     # Print final statistics
#     print("\n" + "=" * 50)
#     print("Training completed!")
#     print(f"Total episodes: {len(episode_rewards)}")
#     print(f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
#     print(f"Max episode reward: {max(episode_rewards)}")
#     print(f"Environment solved: {'Yes' if solved else 'No'}")
    
#     return agent, episode_rewards, avg_rewards

# def test_trained_agent(model_path, episodes=10):
#     """Test a trained agent"""
#     env = gym.make('CartPole-v1', render_mode='human')
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
    
#     # Create agent and load model
#     agent = DQNAgent(state_size, action_size)
#     agent.load(model_path)
#     agent.epsilon = 0  # No exploration during testing
    
#     test_rewards = []
    
#     print("Testing trained agent...")
#     for episode in range(episodes):
#         state, _ = env.reset()
#         total_reward = 0
        
#         while True:
#             action = agent.act(state)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
            
#             state = next_state
#             total_reward += reward
            
#             if done:
#                 break
        
#         test_rewards.append(total_reward)
#         print(f"Test Episode {episode + 1}: Reward = {total_reward}")
    
#     env.close()
    
#     print(f"\nTest Results:")
#     print(f"Average reward: {np.mean(test_rewards):.2f}")
#     print(f"Max reward: {max(test_rewards)}")
#     print(f"Min reward: {min(test_rewards)}")
    
#     return test_rewards

# if __name__ == "__main__":
#     # Train the agent
#     agent, rewards, avg_rewards = train_dqn_agent(episodes=1000)
    
#     # Uncomment to test a trained model
#     # test_rewards = test_trained_agent('dqn_cartpole_final.pth', episodes=5)



import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import deque, namedtuple
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, 
                 batch_size=64, target_update_frequency=100):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.train_start = 1000
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize target network
        self.update_target_network()
        
        # Training metrics
        self.loss_history = []
        self.step_count = 0
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.train_start:
            return
        
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Store loss
        self.loss_history.append(loss.item())
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """Save the model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load the model"""
        checkpoint = torch.load(filename, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        print(f"Model loaded from {filename}")

def plot_training_results(episode_rewards, avg_rewards, loss_history=None, save_path='training_results.png'):
    """Plot comprehensive training results"""
    if loss_history and len(loss_history) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot episode rewards
    ax1.plot(episode_rewards, alpha=0.6, color='blue', linewidth=1, label='Episode Reward')
    ax1.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot average rewards
    ax2.plot(avg_rewards, color='red', linewidth=2, label='Avg Reward (100 episodes)')
    ax2.axhline(y=195, color='green', linestyle='--', linewidth=2, label='Solved (195)')
    ax2.set_title('Average Rewards', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if loss_history and len(loss_history) > 0:
        # Plot training loss
        ax3.plot(loss_history, color='orange', alpha=0.7, linewidth=1)
        ax3.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('MSE Loss')
        ax3.grid(True, alpha=0.3)
        
        # Plot smoothed loss
        window_size = min(100, len(loss_history) // 10)
        if window_size > 1:
            smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
            ax4.plot(smoothed_loss, color='purple', linewidth=2)
            ax4.set_title(f'Smoothed Loss (window={window_size})', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Smoothed MSE Loss')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def train_dqn_agent(episodes=1000, render_frequency=100, save_frequency=200, enable_rendering=True):
    """Train the DQN agent"""
    # Create environments - one for training, one for rendering
    if enable_rendering:
        env = gym.make('CartPole-v1')  # Training environment (no rendering)
        render_env = gym.make('CartPole-v1', render_mode='human')  # Rendering environment
    else:
        env = gym.make('CartPole-v1')
        render_env = None
    
    # Get environment dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Rendering enabled: {enable_rendering}")
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Training metrics
    episode_rewards = []
    avg_rewards = []
    solved = False
    
    print("Starting training...")
    print("=" * 50)
    
    for episode in range(episodes):
        # Decide whether to render this episode
        should_render = (enable_rendering and 
                        (episode % render_frequency == 0 or 
                         episode == episodes - 1 or 
                         (solved and episode % 50 == 0)))
        
        # Choose environment based on rendering
        current_env = render_env if should_render else env
        
        if should_render:
            print(f"🎬 Rendering episode {episode}...")
        
        state, _ = current_env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = current_env.step(action)
            done = terminated or truncated
            
            # Store experience (only if not rendering to avoid bias)
            if not should_render:
                agent.remember(state, action, reward, next_state, done)
                # Train agent
                agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Record metrics
        episode_rewards.append(total_reward)
        
        # Calculate average reward over last 100 episodes
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)
            
            # Check if solved
            if avg_reward >= 195 and not solved:
                print(f"\n🎉 Environment solved in {episode} episodes!")
                print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
                solved = True
                agent.save(f'dqn_cartpole_solved_episode_{episode}.pth')
                
                # Show a celebration render
                if enable_rendering:
                    print("🎊 Celebrating with a demo run!")
                    demo_episode(render_env, agent, episode_num=episode)
        else:
            avg_rewards.append(np.mean(episode_rewards))
        
        # Print progress
        if episode % 50 == 0:
            avg_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            render_status = "🎬" if should_render else ""
            print(f"Episode {episode:4d} {render_status} | Reward: {total_reward:3.0f} | "
                  f"Avg(100): {avg_100:6.2f} | Epsilon: {agent.epsilon:.3f} | "
                  f"Memory: {len(agent.memory):5d}")
        
        # Save model periodically
        if episode % save_frequency == 0 and episode > 0:
            agent.save(f'dqn_cartpole_episode_{episode}.pth')
    
    env.close()
    if render_env:
        render_env.close()
    
    # Save final model
    agent.save('dqn_cartpole_final.pth')
    
    # Plot results
    plot_training_results(episode_rewards, avg_rewards, agent.loss_history)
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Max episode reward: {max(episode_rewards)}")
    print(f"Environment solved: {'Yes' if solved else 'No'}")
    
    return agent, episode_rewards, avg_rewards

def demo_episode(env, agent, episode_num=None, delay=0.02):
    """Run a single demonstration episode with rendering"""
    import time
    
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    print(f"🎮 Running demonstration episode{f' {episode_num}' if episode_num else ''}...")
    
    while True:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        state = next_state
        total_reward += reward
        steps += 1
        
        # Add small delay to make rendering visible
        time.sleep(delay)
        
        if done:
            break
    
    print(f"Demo completed: {total_reward} reward in {steps} steps")
    return total_reward

def test_trained_agent(model_path, episodes=10, enable_rendering=True, delay=0.02):
    """Test a trained agent with optional rendering"""
    if enable_rendering:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
        
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent and load model
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0  # No exploration during testing
    
    test_rewards = []
    
    print(f"Testing trained agent with rendering {'enabled' if enable_rendering else 'disabled'}...")
    print("=" * 60)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"🎮 Test Episode {episode + 1}/{episodes}")
        
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Add delay for better visualization
            if enable_rendering:
                import time
                time.sleep(delay)
            
            if done:
                break
        
        test_rewards.append(total_reward)
        print(f"  ✅ Completed: {total_reward} reward in {steps} steps")
        
        # Small pause between episodes
        if enable_rendering and episode < episodes - 1:
            import time
            time.sleep(1.0)
    
    env.close()
    
    print(f"\nTest Results:")
    print(f"Average reward: {np.mean(test_rewards):.2f}")
    print(f"Max reward: {max(test_rewards)}")
    print(f"Min reward: {min(test_rewards)}")
    
    return test_rewards

def render_best_episodes(agent, num_episodes=5, delay=0.03):
    """Render the best performance episodes of a trained agent"""
    env = gym.make('CartPole-v1', render_mode='human')
    
    print(f"🌟 Showing {num_episodes} demonstration episodes...")
    print("=" * 50)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        reward = demo_episode(env, agent, episode + 1, delay)
        episode_rewards.append(reward)
        
        # Pause between episodes
        if episode < num_episodes - 1:
            import time
            time.sleep(2.0)
    
    env.close()
    
    print("\n🏆 Demo Results:")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Max reward: {max(episode_rewards)}")
    print(f"Min reward: {min(episode_rewards)}")
    
    return episode_rewards

if __name__ == "__main__":
    # Train the agent with rendering
    print("🚀 Starting DQN Training with Rendering")
    print("Settings:")
    print("  - Rendering enabled during training (every 100 episodes)")
    print("  - GPU support:", "✅" if torch.cuda.is_available() else "❌")
    print("  - Target: Solve CartPole (195+ average reward)")
    print()
    
    agent, rewards, avg_rewards = train_dqn_agent(
        episodes=1000, 
        render_frequency=100,  # Render every 100 episodes
        enable_rendering=True
    )
    
    print("\n" + "="*60)
    print("🎉 Training Complete! Now testing the trained agent...")
    
    # Test the final trained model with rendering
    test_rewards = test_trained_agent(
        'dqn_cartpole_final.pth', 
        episodes=5, 
        enable_rendering=True,
        delay=0.02
    )
    
    print(f"\n📊 Final Test Results:")
    print(f"Average test reward: {np.mean(test_rewards):.2f}")
    print(f"Max test reward: {max(test_rewards)}")
    print(f"Min test reward: {min(test_rewards)}")
    
    # Optional: Show additional demo episodes
    print("\n🎬 Want to see more demos? Uncomment the line below!")
    # render_best_episodes(agent, num_episodes=3)