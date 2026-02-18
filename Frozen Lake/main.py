import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

EPISODES = 5000
ALPHA = 0.8
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
SMOOTH = 100

def choose_action(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])

# ─────────────────────────────────────────────
# ALGORITHM 1: Q-Learning (off-policy)
# Uses: max(Q[next_state]) — always picks best
# ─────────────────────────────────────────────
def q_learning(env):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q = np.zeros((n_s, n_a))
    eps = EPSILON
    rewards = []

    for ep in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_r = 0
        while True:
            action = choose_action(Q, state, eps, n_a)
            next_state, reward, done, truncated, _ = env.step(action)

            best_next = np.max(Q[next_state])
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
            state = next_state
            total_r += reward
            if done or truncated:
                break
        eps = max(EPSILON_MIN, eps*EPSILON_DECAY)
        rewards.append(total_r)
    return Q, rewards

# ─────────────────────────────────────────────
# ALGORITHM 2: SARSA (on-policy)
# Uses: Q[next_state][ACTUAL next action taken]
# ─────────────────────────────────────────────
def sarsa(env):
    n_a = env.observation_space.n
    n_s = env.action_space.n
    Q = np.zeros((n_s,n_a))
    eps = EPSILON
    rewards = []

    for ep in range(EPISODES):
        state, _ = env.reset()
        action = choose_action(Q, state, eps, n_a)
        total_r =0
        while True:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = choose_action(Q, next_state, eps, n_a)

            Q[state][action] += ALPHA * (reward + GAMMA * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
            total_r += reward
            if done or truncated:
                break
        eps = max(EPSILON_MIN, eps*EPSILON_DECAY)
        rewards.append(total_r)
    return Q, rewards

# ─────────────────────────────────────────────
# ALGORITHM 3: Expected SARSA
# Uses: AVERAGE of all next actions (weighted by policy)
# ─────────────────────────────────────────────

def expected_sarsa(env):

        