import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List, Optional
from enum import IntEnum


class Action(IntEnum):
    """Enumeration for Pacman actions"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class TileType(IntEnum):
    """Tile type enumeration"""
    EMPTY = 0
    WALL = 1
    FOOD = 2
    POWER_PELLET = 3
    GHOST_SPAWN = 4


class PacmanConfig:
    """Configuration for Pacman environment"""
    TILE_MAP = [
        "XXXXXXXXXXXXXXXXXXX",
        "X        X        X",
        "X XX XXX X XXX XX X",
        "X                 X",
        "X XX X XXXXX X XX X",
        "X    X       X    X",
        "XXXX XXXX XXXX XXXX",
        "OOOX X       X XOOO",
        "XXXX X XXrXX X XXXX",
        "O       bpo       O",
        "XXXX X XXXXX X XXXX",
        "OOOX X       X XOOO",
        "XXXX X XXXXX X XXXX",
        "X        X        X",
        "X XX XXX X XXX XX X",
        "X  X     P     X  X",
        "XX X X XXXXX X X XX",
        "X    X   X   X    X",
        "X XXXXXX X XXXXXX X",
        "X                 X",
        "XXXXXXXXXXXXXXXXXXX"
    ]
    
    ROWS = 21
    COLS = 19
    
    # Rewards
    REWARD_FOOD = 10
    REWARD_POWER_PELLET = 50
    REWARD_GHOST = -50
    REWARD_DEATH = -100
    REWARD_WIN = 500
    REWARD_STEP = -1
    REWARD_WALL_HIT = -5
    
    # Game settings
    INITIAL_LIVES = 3
    GHOST_SPEED = 1  # Steps per ghost move
    
    # Ghost behavior
    GHOST_SCARED_DURATION = 40
    GHOST_SCARED_REWARD = 200


class PacmanEnv(gym.Env):
    """
    Pacman environment following Gymnasium API.
    
    Observation space: 
        - Pacman position (2)
        - Ghost positions (8 for 4 ghosts)
        - Food grid (ROWS * COLS)
        - Additional features: lives, scared timer, score (3)
        
    Action space: 
        - Discrete(4): UP, DOWN, LEFT, RIGHT
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None, use_compact_obs: bool = False):
        super().__init__()
        
        self.config = PacmanConfig()
        self.render_mode = render_mode
        self.use_compact_obs = use_compact_obs
        
        # Action space
        self.action_space = spaces.Discrete(4)
        
        # Observation space
        if use_compact_obs:
            # Compact: positions + distance features
            obs_size = 2 + 8 + 4 + 3  # pac + ghosts + ghost_dists + extras
        else:
            # Full: positions + food grid + extras
            obs_size = 2 + 8 + (self.config.ROWS * self.config.COLS) + 3
            
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize game state
        self._parse_map()
        self.step_count = 0
        
    def _parse_map(self):
        """Parse the tile map and extract initial positions"""
        self.walls = np.zeros((self.config.ROWS, self.config.COLS), dtype=np.uint8)
        self.initial_food = np.zeros((self.config.ROWS, self.config.COLS), dtype=np.uint8)
        self.power_pellets = []
        
        for i, row in enumerate(self.config.TILE_MAP):
            for j, char in enumerate(row):
                if char == 'X':
                    self.walls[i, j] = 1
                elif char == ' ':
                    self.initial_food[i, j] = 1
                elif char == 'P':
                    self.initial_pacman_pos = np.array([i, j])
                    self.initial_food[i, j] = 1
                elif char in 'rbpo':  # Ghost starting positions
                    self.initial_food[i, j] = 1
                elif char == 'O':
                    self.power_pellets.append((i, j))
                    self.initial_food[i, j] = 1
        
        # Find ghost positions (marked with letters in original map)
        self.initial_ghost_positions = []
        ghost_chars = {'r': (8, 11), 'b': (9, 8), 'p': (9, 9), 'o': (9, 10)}
        
        for char, pos in ghost_chars.items():
            for i, row in enumerate(self.config.TILE_MAP):
                if char in row:
                    j = row.index(char)
                    self.initial_ghost_positions.append(np.array([i, j]))
                    break
        
        # Fallback to default positions if not found
        if len(self.initial_ghost_positions) < 4:
            self.initial_ghost_positions = [
                np.array([9, 9]),
                np.array([9, 10]),
                np.array([9, 11]),
                np.array([8, 9])
            ]
        
        self.total_food = np.sum(self.initial_food)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset game state
        self.food = self.initial_food.copy()
        self.pacman_pos = self.initial_pacman_pos.copy()
        self.ghost_positions = [pos.copy() for pos in self.initial_ghost_positions]
        
        self.lives = self.config.INITIAL_LIVES
        self.score = 0
        self.step_count = 0
        self.ghost_scared_timer = 0
        self.done = False
        
        # Ghost movement tracking
        self.ghost_move_counter = 0
        
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        if self.done:
            return self._get_obs(), 0, True, False, self._get_info()
        
        self.step_count += 1
        reward = self.config.REWARD_STEP
        
        # Move Pacman
        reward += self._move_pacman(action)
        
        # Check food collection
        if self.food[self.pacman_pos[0], self.pacman_pos[1]] == 1:
            self.food[self.pacman_pos[0], self.pacman_pos[1]] = 0
            reward += self.config.REWARD_FOOD
            self.score += self.config.REWARD_FOOD
        
        # Check power pellet
        if tuple(self.pacman_pos) in self.power_pellets:
            reward += self.config.REWARD_POWER_PELLET
            self.score += self.config.REWARD_POWER_PELLET
            self.ghost_scared_timer = self.config.GHOST_SCARED_DURATION
            self.power_pellets.remove(tuple(self.pacman_pos))
        
        # Move ghosts
        self.ghost_move_counter += 1
        if self.ghost_move_counter >= self.config.GHOST_SPEED:
            self._move_ghosts()
            self.ghost_move_counter = 0
        
        # Check ghost collision
        ghost_collision = self._check_ghost_collision()
        if ghost_collision:
            if self.ghost_scared_timer > 0:
                reward += self.config.GHOST_SCARED_REWARD
                self.score += self.config.GHOST_SCARED_REWARD
                # Respawn ghost
                self._respawn_ghost(ghost_collision)
            else:
                reward += self.config.REWARD_GHOST
                self.lives -= 1
                if self.lives <= 0:
                    reward += self.config.REWARD_DEATH
                    self.done = True
                else:
                    self._respawn_pacman()
        
        # Update scared timer
        if self.ghost_scared_timer > 0:
            self.ghost_scared_timer -= 1
        
        # Check win condition
        if np.sum(self.food) == 0:
            reward += self.config.REWARD_WIN
            self.score += self.config.REWARD_WIN
            self.done = True
        
        # Timeout check (prevent infinite loops)
        if self.step_count > 1000:
            self.done = True
        
        return self._get_obs(), reward, self.done, False, self._get_info()

    def _move_pacman(self, action: int) -> float:
        """Move Pacman and return reward"""
        directions = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }
        
        dx, dy = directions[action]
        new_pos = self.pacman_pos + np.array([dx, dy])
        
        # Check boundaries
        if not self._is_valid_position(new_pos):
            return self.config.REWARD_WALL_HIT
        
        # Check walls
        if self.walls[new_pos[0], new_pos[1]] == 1:
            return self.config.REWARD_WALL_HIT
        
        # Valid move
        self.pacman_pos = new_pos
        return 0

    def _move_ghosts(self):
        """Simple ghost AI: move randomly towards Pacman"""
        for i, ghost_pos in enumerate(self.ghost_positions):
            # Simple AI: pick direction that gets closer to Pacman
            possible_moves = []
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = ghost_pos + np.array([dx, dy])
                if self._is_valid_position(new_pos) and self.walls[new_pos[0], new_pos[1]] == 0:
                    possible_moves.append(new_pos)
            
            if possible_moves:
                if self.ghost_scared_timer > 0:
                    # Move away from Pacman
                    distances = [np.linalg.norm(pos - self.pacman_pos) for pos in possible_moves]
                    best_move = possible_moves[np.argmax(distances)]
                else:
                    # Move towards Pacman (with some randomness)
                    if np.random.random() < 0.7:
                        distances = [np.linalg.norm(pos - self.pacman_pos) for pos in possible_moves]
                        best_move = possible_moves[np.argmin(distances)]
                    else:
                        best_move = possible_moves[np.random.randint(len(possible_moves))]
                
                self.ghost_positions[i] = best_move

    def _check_ghost_collision(self) -> Optional[int]:
        """Check if Pacman collides with any ghost"""
        for i, ghost_pos in enumerate(self.ghost_positions):
            if np.array_equal(ghost_pos, self.pacman_pos):
                return i
        return None

    def _respawn_pacman(self):
        """Respawn Pacman at initial position"""
        self.pacman_pos = self.initial_pacman_pos.copy()

    def _respawn_ghost(self, ghost_idx: int):
        """Respawn ghost at initial position"""
        self.ghost_positions[ghost_idx] = self.initial_ghost_positions[ghost_idx].copy()

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is within bounds"""
        return (0 <= pos[0] < self.config.ROWS and 
                0 <= pos[1] < self.config.COLS)

    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        # Normalize positions
        pac_normalized = self.pacman_pos / [self.config.ROWS, self.config.COLS]
        ghosts_normalized = np.concatenate([
            g / [self.config.ROWS, self.config.COLS] for g in self.ghost_positions
        ])
        
        if self.use_compact_obs:
            # Compact observation with distance features
            ghost_dists = np.array([
                np.linalg.norm(self.pacman_pos - g) / np.sqrt(self.config.ROWS**2 + self.config.COLS**2)
                for g in self.ghost_positions
            ])
            extras = np.array([
                self.lives / self.config.INITIAL_LIVES,
                min(self.ghost_scared_timer / self.config.GHOST_SCARED_DURATION, 1.0),
                np.sum(self.food) / self.total_food
            ])
            return np.concatenate([pac_normalized, ghosts_normalized, ghost_dists, extras]).astype(np.float32)
        else:
            # Full observation with food grid
            food_flat = self.food.flatten()
            extras = np.array([
                self.lives / self.config.INITIAL_LIVES,
                min(self.ghost_scared_timer / self.config.GHOST_SCARED_DURATION, 1.0),
                np.sum(self.food) / self.total_food
            ])
            return np.concatenate([pac_normalized, ghosts_normalized, food_flat, extras]).astype(np.float32)

    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            'lives': self.lives,
            'score': self.score,
            'food_remaining': int(np.sum(self.food)),
            'step_count': self.step_count,
            'ghost_scared': self.ghost_scared_timer > 0
        }

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            self._render_human()
    
    def _render_human(self):
        """Render to console"""
        display = np.full((self.config.ROWS, self.config.COLS), ' ', dtype=str)
        
        # Draw walls
        display[self.walls == 1] = '█'
        
        # Draw food
        display[self.food == 1] = '·'
        
        # Draw power pellets
        for pellet in self.power_pellets:
            display[pellet] = 'O'
        
        # Draw ghosts
        ghost_char = 'M' if self.ghost_scared_timer > 0 else 'G'
        for ghost in self.ghost_positions:
            display[ghost[0], ghost[1]] = ghost_char
        
        # Draw Pacman
        display[self.pacman_pos[0], self.pacman_pos[1]] = 'P'
        
        # Print
        print('\n' + '='*self.config.COLS)
        for row in display:
            print(''.join(row))
        print(f"Lives: {self.lives} | Score: {self.score} | Food: {np.sum(self.food)}")
        print('='*self.config.ROWS)