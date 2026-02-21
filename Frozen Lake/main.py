from email import policy

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
    n_s = env.observation_space.n   # ✅ states = 16
    n_a = env.action_space.n        # ✅ actions = 4
    Q = np.zeros((n_s, n_a))
    eps = EPSILON
    rewards = []

    for ep in range(EPISODES):
        state, _ = env.reset()
        action = choose_action(Q, state, eps, n_a)
        total_r = 0

        while True:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = choose_action(Q, next_state, eps, n_a)

            Q[state][action] += ALPHA * (
                reward + GAMMA * Q[next_state][next_action]
                - Q[state][action]
            )

            state = next_state
            action = next_action
            total_r += reward

            if done or truncated:
                break

        eps = max(EPSILON_MIN, eps * EPSILON_DECAY)
        rewards.append(total_r)

    return Q, rewards

# ─────────────────────────────────────────────
# ALGORITHM 3: Expected SARSA
# Uses: AVERAGE of all next actions (weighted by policy)
# ─────────────────────────────────────────────

def expected_sarsa(env):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q = np.zeros((n_s,n_a))
    eps = EPSILON
    rewards = []
    for ep in range(EPISODES):
        state, _ = env.reset()
        total_r = 0

        while True:
            action = choose_action(Q, state, eps, n_a)
            next_state, reward, done, truncated, _ = env.step(action)
            policy = np.ones(n_a) * (eps / n_a)
            best_a = np.argmax(Q[next_state])
            policy[best_a] += (1 - eps)
            best_a = np.argmax(Q[next_state])
            policy[best_a] += (1-eps)
            expected_q = np.dot(policy, Q[next_state])
            Q[state][action] += ALPHA *(reward + GAMMA *expected_q - Q[state][action])
            state = next_state
            total_r += reward
            if done or truncated:
                break
        eps = max(EPSILON_MIN, eps*EPSILON_DECAY)
        rewards.append(total_r)
    return Q, rewards
# ─────────────────────────────────────────────
# PLOT 1: Reward Curves (smoothed)
# ─────────────────────────────────────────────
def smooth(data, window=SMOOTH):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_rewards(ql_r, sarsa_r, esarsa_r):
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(ql_r),     label='Q-Learning',      color='#63b3ed', lw=2)
    plt.plot(smooth(sarsa_r),  label='SARSA',           color='#f6ad55', lw=2)
    plt.plot(smooth(esarsa_r), label='Expected SARSA',  color='#68d391', lw=2)
    plt.xlabel('Episode')
    plt.ylabel(f'Reward (smoothed over {SMOOTH} eps)')
    plt.title('🧊 FrozenLake: Algorithm Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_curves.png', dpi=150)
    plt.show()
    print("✅ Saved: reward_curves.png")


# ─────────────────────────────────────────────
# PLOT 2: Q-Table Heatmap (best action values)
# ─────────────────────────────────────────────
def plot_qtable(Q, title='Q-Table'):
    # Max Q-value per state, reshaped to 4x4 grid
    best_q = np.max(Q, axis=1).reshape(4, 4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: max Q-value heatmap
    im = axes[0].imshow(best_q, cmap='Blues')
    axes[0].set_title(f'{title}: Best Q-Values')
    plt.colorbar(im, ax=axes[0])
    # Label holes/goal
    labels = ['S','F','F','F', 'F','H','F','H',
              'F','F','F','H', 'H','F','F','G']
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, labels[i*4+j],
                         ha='center', va='center', fontsize=14, color='red')

    # Right: best action arrows
    arrow_map = {0:'←', 1:'↓', 2:'→', 3:'↑'}
    best_a = np.argmax(Q, axis=1).reshape(4, 4)
    axes[1].imshow(best_q, cmap='Greens', alpha=0.4)
    axes[1].set_title(f'{title}: Best Actions')
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, arrow_map[best_a[i,j]],
                         ha='center', va='center', fontsize=20)

    plt.tight_layout()
    plt.savefig('qtable_heatmap.png', dpi=150)
    plt.show()
    print("✅ Saved: qtable_heatmap.png")


# MAIN - run Everything

if __name__ == "__main__":
    # is_slippery=True makes it stochastic (harder!)
    env = gym.make('FrozenLake-v1', is_slippery=True)

    print("🏃 Training Q-Learning...")
    Q_ql, r_ql = q_learning(env)

    print("🏃 Training SARSA...")
    Q_sa, r_sa = sarsa(env)

    print("🏃 Training Expected SARSA...")
    Q_es, r_es = expected_sarsa(env)

    # Print final win rates
    for name, r in [('Q-Learning', r_ql), ('SARSA', r_sa), ('Exp SARSA', r_es)]:
        win_rate = np.mean(r[-500:]) * 100
        print(f"  {name}: {win_rate:.1f}% win rate (last 500 eps)")

    plot_rewards(r_ql, r_sa, r_es)
    plot_qtable(Q_ql, title='Q-Learning')

    env.close()
