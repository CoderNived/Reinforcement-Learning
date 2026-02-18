import random
import numpy as np
import gymnasium as gym

env = gym.make('Taxi-v3')
alpha = 0.9
gamma=0.95
epsilon=0.1
epsilon_decay=0.9995
min_epsilon=0.01
num_episodes=10000
max_steps=100

q_table = np.zeros((env.observation_space.n, env.action_space.n))

# def choose_action(state):
#     if random.uniform(a:0, b:1)<epsilon:
#     return env.action_space.sample()
#     else:
#     return env.action_space.sample()

def choose_action(state):
        if random.uniform(0,1)<epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(q_table[state, :])
for episode in range(num_episodes):
     state, _ = env.reset()
     done=False
     for steps in range(max_steps):
          action=choose_action(state)
          next_state, reward, done, truncated, info=env.step(action)
          old_value = q_table[state,action]
          next_max=np.max(q_table[next_state,:])

          q_table[state,action]=(1-alpha)*old_value+alpha*(reward+gamma*next_max)
          state=next_state

          if done or truncated:
               break
          epsilon = max(min_epsilon, epsilon*epsilon_decay)
env = gym.make(id= 'Taxi-v3', render_mode='human')

for episode in range(5):
     state, _=env.reset()
     done=False
     print('Episode',episode)
     for step in range(max_steps):
          env.render()
          action=np.argmax(q_table[state, :])
          next_state, reward, done, truncated, info = env.step(action)
          state=next_state

          if done or truncated:
               env.render()
               print('Finished Episode', episode, 'with reward', reward)
               break
          
env.close()
