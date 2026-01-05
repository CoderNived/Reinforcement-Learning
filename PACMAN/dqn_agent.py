import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DQNConfig:
    """Configuration for DQN agent"""
    state_dim: int
    action_dim: int
    hidden_dims: List[int] = None
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    memory_size: int = 100_000
    batch_size: int = 64
    target_update_freq: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class DQN(nn.Module):
    """Deep Q-Network with flexible architecture"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer with efficient numpy storage"""
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of experiences"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )
    
    def __len__(self):
        return self.size


class DQNAgent:
    """Deep Q-Learning agent with Double DQN and improved training"""
    
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.model = DQN(
            config.state_dim, 
            config.action_dim, 
            config.hidden_dims
        ).to(self.device)
        
        self.target = DQN(
            config.state_dim, 
            config.action_dim, 
            config.hidden_dims
        ).to(self.device)
        
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Replay buffer
        self.memory = ReplayBuffer(config.memory_size, config.state_dim)
        
        # Exploration
        self.epsilon = config.epsilon_start
        
        # Training metrics
        self.training_steps = 0
        self.losses = []
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.config.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax(1).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
    
    def train(self) -> Optional[float]:
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.config.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: use online network for action selection, target for evaluation
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # Compute loss with Huber loss for stability
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.config.target_update_freq == 0:
            self.update_target()
        
        # Track loss
        loss_val = loss.item()
        self.losses.append(loss_val)
        
        return loss_val
    
    def update_target(self):
        """Update target network with current model weights"""
        self.target.load_state_dict(self.model.state_dict())
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
    
    def get_metrics(self) -> dict:
        """Get training metrics"""
        return {
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0
        }


# Example usage
if __name__ == "__main__":
    config = DQNConfig(
        state_dim=4,
        action_dim=4,
        hidden_dims=[128, 64],
        learning_rate=1e-3
    )
    
    agent = DQNAgent(config)
    
    # Training loop example
    for episode in range(10):
        state = np.random.randn(4)
        done = False
        
        while not done:
            action = agent.act(state)
            next_state = np.random.randn(4)
            reward = np.random.randn()
            done = np.random.random() > 0.95
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()
            
            state = next_state
        
        print(f"Episode {episode}, Metrics: {agent.get_metrics()}")