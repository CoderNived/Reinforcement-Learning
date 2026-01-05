import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
from collections import deque

from pacman_env import PacmanEnv
from dqn_agent import DQNAgent, DQNConfig


class TrainingMetrics:
    """Track and analyze training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.episode_foods = []
        self.losses = []
        self.epsilon_history = []
        
        self.window_size = window_size
        self.best_reward = -float('inf')
        self.best_score = 0
    
    def update(self, reward: float, length: int, score: int, food: int, 
               epsilon: float, loss: float = None):
        """Update metrics for current episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_scores.append(score)
        self.episode_foods.append(food)
        self.epsilon_history.append(epsilon)
        
        if loss is not None:
            self.losses.append(loss)
        
        self.best_reward = max(self.best_reward, reward)
        self.best_score = max(self.best_score, score)
    
    def get_recent_stats(self) -> Dict:
        """Get statistics for recent episodes"""
        if len(self.episode_rewards) == 0:
            return {}
        
        n = min(self.window_size, len(self.episode_rewards))
        
        return {
            'avg_reward': np.mean(self.episode_rewards[-n:]),
            'avg_length': np.mean(self.episode_lengths[-n:]),
            'avg_score': np.mean(self.episode_scores[-n:]),
            'avg_food': np.mean(self.episode_foods[-n:]),
            'best_reward': self.best_reward,
            'best_score': self.best_score,
            'current_epsilon': self.epsilon_history[-1] if self.epsilon_history else 1.0,
            'avg_loss': np.mean(self.losses[-n:]) if self.losses else 0
        }
    
    def plot_training(self, save_path: str = None):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Smooth data for better visualization
        def smooth(data, weight=0.9):
            smoothed = []
            last = data[0] if data else 0
            for point in data:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed
        
        # Plot rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) > 10:
            axes[0, 0].plot(smooth(self.episode_rewards), label='Smoothed')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot scores
        axes[0, 1].plot(self.episode_scores, alpha=0.3, label='Raw')
        if len(self.episode_scores) > 10:
            axes[0, 1].plot(smooth(self.episode_scores), label='Smoothed')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Episode Scores')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot epsilon
        axes[1, 0].plot(self.epsilon_history)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate')
        axes[1, 0].grid(True)
        
        # Plot food collected
        axes[1, 1].plot(self.episode_foods, alpha=0.3, label='Raw')
        if len(self.episode_foods) > 10:
            axes[1, 1].plot(smooth(self.episode_foods), label='Smoothed')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Food Collected')
        axes[1, 1].set_title('Food Collection Progress')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Training plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class PacmanTrainer:
    """Training manager for Pacman DQN agent"""
    
    def __init__(self, 
                 episodes: int = 2000,
                 eval_frequency: int = 50,
                 eval_episodes: int = 5,
                 save_frequency: int = 100,
                 render_eval: bool = False,
                 use_compact_obs: bool = False,
                 checkpoint_dir: str = 'checkpoints'):
        
        self.episodes = episodes
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.save_frequency = save_frequency
        self.render_eval = render_eval
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize environment and agent
        self.env = PacmanEnv(use_compact_obs=use_compact_obs)
        self.eval_env = PacmanEnv(
            render_mode='human' if render_eval else None,
            use_compact_obs=use_compact_obs
        )
        
        # Configure agent
        config = DQNConfig(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_dims=[256, 128, 64],
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=50_000,
            batch_size=64,
            target_update_freq=10
        )
        
        self.agent = DQNAgent(config)
        self.metrics = TrainingMetrics(window_size=100)
        
        # Training state
        self.start_time = datetime.now()
        self.total_steps = 0
    
    def train_episode(self) -> Dict:
        """Train for one episode"""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_loss = []
        steps = 0
        
        while True:
            # Select and perform action
            action = self.agent.act(state, training=True)
            next_state, reward, done, _, info = self.env.step(action)
            
            # Store transition
            self.agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = self.agent.train()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            steps += 1
            self.total_steps += 1
            
            if done:
                break
        
        return {
            'reward': episode_reward,
            'length': steps,
            'score': info['score'],
            'food': self.env.total_food - info['food_remaining'],
            'loss': np.mean(episode_loss) if episode_loss else None
        }
    
    def evaluate(self) -> Dict:
        """Evaluate agent performance"""
        eval_rewards = []
        eval_scores = []
        eval_foods = []
        
        for _ in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.act(state, training=False)
                next_state, reward, done, _, info = self.eval_env.step(action)
                
                if self.render_eval:
                    self.eval_env.render()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    eval_rewards.append(episode_reward)
                    eval_scores.append(info['score'])
                    eval_foods.append(self.env.total_food - info['food_remaining'])
                    break
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_score': np.mean(eval_scores),
            'avg_food': np.mean(eval_foods),
            'std_reward': np.std(eval_rewards)
        }
    
    def save_checkpoint(self, episode: int, eval_stats: Dict = None):
        """Save agent checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"agent_ep{episode}_{timestamp}.pt"
        
        self.agent.save(str(checkpoint_path))
        
        # Save training metrics
        metrics_path = self.checkpoint_dir / f"metrics_ep{episode}_{timestamp}.json"
        metrics_data = {
            'episode': episode,
            'total_steps': self.total_steps,
            'training_stats': self.metrics.get_recent_stats(),
            'eval_stats': eval_stats,
            'timestamp': timestamp
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("="*70)
        print("Starting Pacman DQN Training")
        print(f"Episodes: {self.episodes}")
        print(f"Environment: {self.env.config.ROWS}x{self.env.config.COLS}")
        print(f"Observation dim: {self.env.observation_space.shape[0]}")
        print(f"Action dim: {self.env.action_space.n}")
        print(f"Device: {self.agent.device}")
        print("="*70)
        
        try:
            for episode in range(self.episodes):
                # Train one episode
                ep_stats = self.train_episode()
                
                # Update metrics
                self.metrics.update(
                    reward=ep_stats['reward'],
                    length=ep_stats['length'],
                    score=ep_stats['score'],
                    food=ep_stats['food'],
                    epsilon=self.agent.epsilon,
                    loss=ep_stats['loss']
                )
                
                # Print progress
                if episode % 10 == 0:
                    stats = self.metrics.get_recent_stats()
                    print(f"Ep {episode:4d} | "
                          f"R: {ep_stats['reward']:6.1f} | "
                          f"Avg R: {stats['avg_reward']:6.1f} | "
                          f"Score: {ep_stats['score']:4d} | "
                          f"Food: {ep_stats['food']:3d} | "
                          f"ε: {self.agent.epsilon:.3f} | "
                          f"Steps: {self.total_steps:6d}")
                
                # Evaluate
                if episode > 0 and episode % self.eval_frequency == 0:
                    print("\n" + "-"*70)
                    print(f"Evaluating at episode {episode}...")
                    eval_stats = self.evaluate()
                    print(f"Eval Results: "
                          f"Avg Reward: {eval_stats['avg_reward']:.1f} ± {eval_stats['std_reward']:.1f} | "
                          f"Avg Score: {eval_stats['avg_score']:.1f} | "
                          f"Avg Food: {eval_stats['avg_food']:.1f}")
                    print("-"*70 + "\n")
                else:
                    eval_stats = None
                
                # Save checkpoint
                if episode > 0 and episode % self.save_frequency == 0:
                    self.save_checkpoint(episode, eval_stats)
                    
                    # Save training plot
                    plot_path = self.checkpoint_dir / f"training_progress_ep{episode}.png"
                    self.metrics.plot_training(str(plot_path))
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        
        finally:
            # Final evaluation and save
            print("\n" + "="*70)
            print("Training completed!")
            print(f"Total episodes: {episode + 1}")
            print(f"Total steps: {self.total_steps}")
            print(f"Training time: {datetime.now() - self.start_time}")
            
            final_stats = self.metrics.get_recent_stats()
            print(f"\nFinal Stats:")
            print(f"  Best reward: {final_stats['best_reward']:.1f}")
            print(f"  Best score: {final_stats['best_score']}")
            print(f"  Avg reward (last 100): {final_stats['avg_reward']:.1f}")
            print(f"  Avg food collected: {final_stats['avg_food']:.1f}")
            
            # Final evaluation
            print("\nRunning final evaluation...")
            final_eval = self.evaluate()
            print(f"Final Eval: Reward={final_eval['avg_reward']:.1f}, "
                  f"Score={final_eval['avg_score']:.1f}")
            
            # Save final checkpoint
            self.save_checkpoint(episode + 1, final_eval)
            
            # Save final plot
            plot_path = self.checkpoint_dir / "training_final.png"
            self.metrics.plot_training(str(plot_path))
            
            print("="*70)


def main():
    """Main entry point"""
    trainer = PacmanTrainer(
        episodes=2000,
        eval_frequency=50,
        eval_episodes=5,
        save_frequency=100,
        render_eval=False,  # Set True to watch evaluation
        use_compact_obs=False,  # Use compact observations for faster training
        checkpoint_dir='pacman_checkpoints'
    )
    
    trainer.train()


if __name__ == "__main__":
    main()