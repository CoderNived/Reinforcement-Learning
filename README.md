# 🤖 Reinforcement Learning: A Project-Based Guide

> Train an agent to balance a pole on a cart using Deep Q-Learning (DQN)

---

## 📚 What is Reinforcement Learning?

RL is about training an **agent** to make decisions by interacting with an **environment**. The agent receives rewards or penalties based on its actions and learns to maximize cumulative reward over time.

### Core Components

| Component | Description |
|-----------|-------------|
| **Agent** | The learner / decision maker |
| **Environment** | What the agent interacts with |
| **State (S)** | The current situation |
| **Action (A)** | What the agent can do |
| **Reward (R)** | Feedback signal |
| **Policy (π)** | The agent's strategy (state → action) |

### The RL Loop

```
Agent → takes Action → Environment
Environment → returns new State + Reward → Agent
Agent → updates Policy → repeat
```

---

## 🧠 Key Concepts

### 1. The Bellman Equation

The foundation of most RL algorithms. The value of a state equals the immediate reward plus the discounted value of the next state:

```
Q(s, a) = r + γ * max(Q(s', a'))
```

- **γ (gamma)** — discount factor (0–1), how much future rewards matter
- **Q(s, a)** — expected return from taking action `a` in state `s`

### 2. Exploration vs. Exploitation

- **Explore** — try new actions to discover better rewards
- **Exploit** — use known good actions
- **ε-greedy** — the common strategy: explore with probability ε, exploit otherwise

---

## 🚀 Project: Train an Agent to Play CartPole

Balance a pole on a moving cart — the classic RL beginner project.

### Setup

```bash
pip install gymnasium numpy torch
```

---

### Step 1: Understand the Environment

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

# obs = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
print("Observation space:", env.observation_space)  # 4 continuous values
print("Action space:", env.action_space)             # 2 actions: push left or right
```

---

### Step 2: Build a Q-Network (DQN)

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

state_dim = 4   # CartPole observation size
action_dim = 2  # left or right
model = QNetwork(state_dim, action_dim)
```

---

### Step 3: Replay Buffer

Store experiences and sample random batches to break correlation between samples:

```python
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)
```

---

### Step 4: Training Loop

```python
import torch.optim as optim

# Hyperparameters
GAMMA         = 0.99
LR            = 1e-3
BATCH_SIZE    = 64
EPSILON_START = 1.0
EPSILON_END   = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10   # update target network every N episodes

env = gym.make("CartPole-v1")
q_net      = QNetwork(4, 2)
target_net = QNetwork(4, 2)
target_net.load_state_dict(q_net.state_dict())  # same weights initially

optimizer = optim.Adam(q_net.parameters(), lr=LR)
buffer    = ReplayBuffer()
epsilon   = EPSILON_START

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0

    while True:
        # ε-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()           # explore
        else:
            with torch.no_grad():
                q_values = q_net(torch.FloatTensor(state))
                action = q_values.argmax().item()        # exploit

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train when buffer has enough samples
        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            states_t      = torch.FloatTensor(states)
            actions_t     = torch.LongTensor(actions)
            rewards_t     = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t       = torch.FloatTensor(dones)

            # Current Q values
            current_q = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

            # Target Q values (using frozen target network)
            with torch.no_grad():
                max_next_q = target_net(next_states_t).max(1)[0]
                target_q   = rewards_t + GAMMA * max_next_q * (1 - dones_t)

            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.0f}, Epsilon: {epsilon:.3f}")
```

---

### Step 5: Evaluate Your Agent

```python
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()

for _ in range(1000):
    with torch.no_grad():
        action = q_net(torch.FloatTensor(state)).argmax().item()
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        state, _ = env.reset()
```

---

## ✅ What You've Learned

| Concept | Where It Appeared |
|---|---|
| Agent & Environment | CartPole setup |
| State, Action, Reward | `env.step()` returns |
| Q-Learning (Bellman) | Loss function |
| ε-greedy exploration | Action selection |
| Experience Replay | `ReplayBuffer` |
| Target Network | Stable training |

---

## 🔭 Next Steps

- Try other environments — `LunarLander-v2`, `MountainCar-v0`, Atari games
- Learn Policy Gradient methods — REINFORCE, Actor-Critic (A2C)
- Explore **PPO** (Proximal Policy Optimization) — the industry workhorse
- Use **Stable Baselines3** — pre-built RL algorithms library
- Dive into **Multi-agent RL** — agents that interact with each other