import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Hyperparameters
# ==============================
EPISODES = 5000
ALPHA = 0.8
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
SMOOTH = 100


# ==============================
# Epsilon-Greedy Policy
# ==============================
def choose_action(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])


# ==============================
# Q-Learning (Off-policy)
# ==============================
def q_learning(env):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q = np.zeros((n_s, n_a))
    eps = EPSILON
    rewards = []

    for ep in range(EPISODES):
        state, _ = env.reset()
        total_r = 0

        while True:
            action = choose_action(Q, state, eps, n_a)
            next_state, reward, done, truncated, _ = env.step(action)

            best_next = np.max(Q[next_state])
            Q[state, action] += ALPHA * (
                reward + GAMMA * best_next - Q[state, action]
            )

            state = next_state
            total_r += reward

            if done or truncated:
                break

        eps = max(EPSILON_MIN, eps * EPSILON_DECAY)
        rewards.append(total_r)

    return Q, rewards


# ==============================
# SARSA (On-policy)
# ==============================
def sarsa(env):
    n_s = env.observation_space.n
    n_a = env.action_space.n
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

            Q[state, action] += ALPHA * (
                reward + GAMMA * Q[next_state, next_action] - Q[state, action]
            )

            state = next_state
            action = next_action
            total_r += reward

            if done or truncated:
                break

        eps = max(EPSILON_MIN, eps * EPSILON_DECAY)
        rewards.append(total_r)

    return Q, rewards


# ==============================
# Expected SARSA
# ==============================
def expected_sarsa(env):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q = np.zeros((n_s, n_a))
    eps = EPSILON
    rewards = []

    for ep in range(EPISODES):
        state, _ = env.reset()
        total_r = 0

        while True:
            action = choose_action(Q, state, eps, n_a)
            next_state, reward, done, truncated, _ = env.step(action)

            # Build epsilon-greedy distribution
            policy = np.ones(n_a) * (eps / n_a)
            best_action = np.argmax(Q[next_state])
            policy[best_action] += (1 - eps)

            expected_q = np.dot(policy, Q[next_state])

            Q[state, action] += ALPHA * (
                reward + GAMMA * expected_q - Q[state, action]
            )

            state = next_state
            total_r += reward

            if done or truncated:
                break

        eps = max(EPSILON_MIN, eps * EPSILON_DECAY)
        rewards.append(total_r)

    return Q, rewards


# ==============================
# Plot Utilities
# ==============================
def smooth(data, window=SMOOTH):
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_rewards(ql_r, sarsa_r, esarsa_r):
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(ql_r), label='Q-Learning', lw=2)
    plt.plot(smooth(sarsa_r), label='SARSA', lw=2)
    plt.plot(smooth(esarsa_r), label='Expected SARSA', lw=2)

    plt.xlabel('Episode')
    plt.ylabel(f'Reward (smoothed over {SMOOTH})')
    plt.title('FrozenLake: Algorithm Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_qtable(Q, title='Q-Table'):
    best_q = np.max(Q, axis=1).reshape(4, 4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im = axes[0].imshow(best_q, cmap='Blues')
    axes[0].set_title(f'{title}: Best Q-Values')
    plt.colorbar(im, ax=axes[0])

    labels = ['S','F','F','F',
              'F','H','F','H',
              'F','F','F','H',
              'H','F','F','G']

    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, labels[i*4+j],
                         ha='center', va='center',
                         fontsize=14, color='red')

    arrow_map = {0:'←', 1:'↓', 2:'→', 3:'↑'}
    best_a = np.argmax(Q, axis=1).reshape(4, 4)

    axes[1].imshow(best_q, cmap='Greens', alpha=0.4)
    axes[1].set_title(f'{title}: Best Actions')

    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, arrow_map[best_a[i,j]],
                         ha='center', va='center',
                         fontsize=20)

    plt.tight_layout()
    plt.show()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    # Separate environments (important)
    env_ql = gym.make('FrozenLake-v1', is_slippery=True)
    env_sa = gym.make('FrozenLake-v1', is_slippery=True)
    env_es = gym.make('FrozenLake-v1', is_slippery=True)

    print("Training Q-Learning...")
    Q_ql, r_ql = q_learning(env_ql)

    print("Training SARSA...")
    Q_sa, r_sa = sarsa(env_sa)

    print("Training Expected SARSA...")
    Q_es, r_es = expected_sarsa(env_es)

    # Win rates
    for name, r in [('Q-Learning', r_ql),
                    ('SARSA', r_sa),
                    ('Expected SARSA', r_es)]:

        win_rate = np.mean(r[-500:]) * 100
        print(f"{name}: {win_rate:.2f}% win rate (last 500 episodes)")

    plot_rewards(r_ql, r_sa, r_es)
    plot_qtable(Q_ql, title='Q-Learning')

    env_ql.close()
    env_sa.close()
    env_es.close()


    # ==========================
    # Render trained Q-Learning
    # ==========================
    env_render = gym.make(
        'FrozenLake-v1',
        is_slippery=True,
        render_mode='human'
    )

    print("\nRunning trained Q-Learning policy...\n")

    for game in range(10):
        state, _ = env_render.reset()
        print(f"\n--- Game {game+1} ---")

        while True:
            action = np.argmax(Q_ql[state])
            state, reward, done, truncated, _ = env_render.step(action)

            if done or truncated:
                print("Won!" if reward > 0 else "Fell in hole")
                break

    env_render.close()