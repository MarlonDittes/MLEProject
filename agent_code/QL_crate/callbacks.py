import os
import pickle
import random

import numpy as np

from random import shuffle
from collections import defaultdict

import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Hyperparameters
    self.alpha = 0.1   # Learning rate
    self.gamma = 0.9  # Discount factor

    self.episodes = 4400

    # Epsilon Decay Parameters
    self.epsilon_start = 1    # Initial exploration rate
    self.epsilon_min = 0.01     # Minimum exploration rate
    self.epsilon_decay_rate = 0.00025  # How much to decay epsilon per episode
    self.epsilon = self.epsilon_start  # Initialize epsilon

    # Track episodes
    self.episode_number = 0

    # Plotting
    self.td_error = []
    self.rewards = []
    self.epsilon_history = []

    self.episode_td_error = []
    self.episode_rewards = []

    # Train from new start
    self.reset = False

    # Setup
    if not os.path.isfile("q_table.pt") or (self.train and self.reset):
        self.logger.info("Setting up model from scratch.")
        self.q_table = defaultdict(default_action_probabilities)
    else:
        self.logger.info("Loading model from saved state.")
        with open("q_table.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Update epsilon with decay
    if self.train:
        self.epsilon = max(self.epsilon_min, self.epsilon_start - self.epsilon_decay_rate * self.episode_number)
        self.logger.debug(f"Epsilon: {self.epsilon}")

    # Epsilon-greedy action selection
    if self.train and random.uniform(0, 1) < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    state = tuple(state_to_features(game_state))
    self.logger.debug(f"State: {state}")
    self.logger.debug(f"Action: {ACTIONS[np.argmax(self.q_table[state])]}")
    return ACTIONS[np.argmax(self.q_table[state])]


def state_to_features(game_state: dict, logger=None) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None
    
    self = game_state["self"][3]
    arena = game_state['field']
    coins = game_state["coins"]
    explosion_map = game_state["explosion_map"]
    bombs = game_state["bombs"]

    # setup free space for finding paths
    free_space = game_state["field"] == 0
    others_pos = [xy for (n, s, b, xy) in game_state['others']]
    for pos in others_pos:
        free_space[pos] = False
    bomb_pos = [xy for (xy, t) in game_state["bombs"]]
    for pos in bomb_pos:
        free_space[pos] = False

    # direction of nearest coin
    nearest_coin_direction = look_for_targets(free_space, self, coins, logger)

    if nearest_coin_direction is not None:
        to_coin = tuple(a - b for a,b in zip(nearest_coin_direction, self))
    else:
        to_coin = (0,0)

    
    # direction of nearest dead end
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]

    nearest_dead_end_direction = look_for_targets(free_space, self, dead_ends, logger)
    if nearest_dead_end_direction is not None:
        to_dead_end = tuple(a - b for a,b in zip(nearest_dead_end_direction, self))
    else:
        to_dead_end = (0,0)

    # direction of nearest safe tile
    bomb_positions = [pos for pos, _ in game_state['bombs']]
    explosion_positions = np.argwhere(explosion_map == 1)

    def is_path_blocked(start, end, arena):
        """Check if the path from start to end is blocked by walls."""
        sx, sy = start
        ex, ey = end
        if sx == ex:  # Vertical movement
            for y in range(min(sy, ey) + 1, max(sy, ey)):
                if arena[sx, y] == -1:
                    return True
        elif sy == ey:  # Horizontal movement
            for x in range(min(sx, ex) + 1, max(sx, ex)):
                if arena[x, sy] == -1:
                    return True
        return False
    
    to_safety = (0,0)
    danger_tiles = set()
    for ex, ey in explosion_positions:
        danger_tiles.add((ex, ey))
    if len(bomb_positions) > 0:
        for bx, by in bomb_positions:
            danger_tiles.add((bx, by))

            # Check tiles in the same row
            for dx in range(1, s.BOMB_POWER + 1):
                if bx + dx < s.ROWS-1 and not is_path_blocked((bx, by), (bx + dx, by), arena):
                    danger_tiles.add((bx + dx, by))
                if bx - dx >= 1 and not is_path_blocked((bx, by), (bx - dx, by), arena):
                    danger_tiles.add((bx - dx, by))
            
            # Check tiles in the same column
            for dy in range(1, s.BOMB_POWER + 1):
                if by + dy < s.COLS-1 and not is_path_blocked((bx, by), (bx, by + dy), arena):
                    danger_tiles.add((bx, by + dy))
                if by - dy >= 1 and not is_path_blocked((bx, by), (bx, by - dy), arena):
                    danger_tiles.add((bx, by - dy))

        all_tiles = set((x, y) for x in range(1, s.ROWS-1) for y in range(1, s.COLS-1))
        safe_tiles = all_tiles - danger_tiles
        free_tiles = set((x, y) for x in range(1, s.ROWS-1) for y in range(1, s.COLS-1) if free_space[x, y] == True)
        escape_tiles = safe_tiles & free_tiles

        nearest_escape_direction = look_for_targets(free_space, self, list(escape_tiles), logger)
        if nearest_dead_end_direction is not None:
            to_safety = tuple(a - b for a,b in zip(nearest_escape_direction, self))
        else:
            to_safety = (0,0)

    # is there imminent danger in any of the directions
    danger = [False, False, False, False]  # UP, RIGHT, DOWN, LEFT
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    for i, (dx, dy) in enumerate(directions):
        nx, ny = self[0] + dx, self[1] + dy
        
        if not (1 <= nx < s.ROWS-1 and 1 <= ny < s.COLS-1) or arena[nx, ny] == -1:
            continue

        # Check if there is an explosion
        if explosion_map[nx, ny] == 1:
            danger[i] = True
        else:
            # Check if there is a bomb with countdown == 0
            for (bx, by), countdown in bombs:
                if countdown == 0 and ((bx == nx and abs(by - ny) <= s.BOMB_POWER) or (by == ny and abs(bx - nx) <= s.BOMB_POWER)):
                    if not is_path_blocked((bx, by), (nx, ny), arena):
                        danger[i] = True

    # have bomb?
    have_bomb = game_state['self'][2]

    #TODO: use these features?
    # bomb value
    adjacent_positions = [
        (self[0] + 1, self[1]),  
        (self[0] - 1, self[1]),  
        (self[0], self[1] + 1),  
        (self[0], self[1] - 1)   
    ]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    adjacent_crates = [pos for pos in adjacent_positions if pos in crates]

    bomb_value = len(adjacent_crates)

    # distance to nearest coin
    distances = [np.abs(x - self[0]) + np.abs(y - self[1]) for (x, y) in coins]
    if distances:
        min_distance_to_coin = min(distances)
    #TODO: end

    # build feature array
    features = np.array(
        list(to_coin) + list(to_dead_end) + list(to_safety) + [have_bomb] + list(danger),
        dtype=np.float32
    )

    return features

def default_action_probabilities():
    weights = np.random.rand(len(ACTIONS))
    return weights / weights.sum()

    #weights = np.zeros(len(ACTIONS))
    #return weights

def save_metrics(self, filename='metrics.pkl'):
    """Save the metrics to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump({
            'td_error': self.td_error,
            'rewards': self.rewards,
            'epsilon_history': self.epsilon_history
        }, f)