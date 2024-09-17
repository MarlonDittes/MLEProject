import os
import pickle
import random
import numpy as np
from .model import DQNAgent
from random import shuffle
from scipy.spatial import distance

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
    Initializes the DQNAgent for training or playing.
    """
    # Define the input and output dimensions for the DQNAgent
    input_dim = 15
    output_dim = len(ACTIONS)

    # Initialize the DQN agent
    self.agent = DQNAgent(state_size=input_dim, action_size=output_dim)

    # Load the model if available and not in training mode
    if not self.train and os.path.isfile("my-saved-model.pt"):
        self.logger.info("Loading model from saved state.")
        self.agent.load("my-saved-model.pt")
    else:
        self.logger.info("Setting up model from scratch.")


def act(self, game_state: dict) -> str:
    """
    The agent selects an action using epsilon-greedy strategy via the DQN agent.
    """
    # Convert game state to feature vector
    state = state_to_features(game_state)

    # Get the action from the DQN agent
    action_index = self.agent.act(state)

    # Return the action corresponding to the chosen index
    return ACTIONS[action_index]


def state_to_features(game_state: dict) -> np.array:
    """
    Convert the game state into a feature vector for the DQN.
    """
    if game_state is None:
        # If no game state is available, return a zeroed feature vector of the expected size
        return np.zeros(self.model.state_size, dtype=np.float32)
    
    field = game_state["field"]
    free_space = field == 0
    
    # Positions of other agents
    others_pos = [xy for (n, s, b, xy) in game_state['others']]
    for pos in others_pos:
        free_space[pos] = False
    
    # Positions of bombs
    bomb_pos = [xy for (xy, t) in game_state["bombs"]]
    for pos in bomb_pos:
        free_space[pos] = False

    x, y = game_state["self"][3]  # Current position of the agent

    # Feature 1: Check if moving in each direction is possible
    def can_move(x, y, direction):
        if direction == 'UP':
            return free_space[x, y - 1]
        elif direction == 'DOWN':
            return free_space[x, y + 1] 
        elif direction == 'LEFT':
            return free_space[x - 1, y]
        elif direction == 'RIGHT':
            return free_space[x + 1, y]
        return False

    up_possible = can_move(x, y, 'UP')
    down_possible = can_move(x, y, 'DOWN')
    left_possible = can_move(x, y, 'LEFT')
    right_possible = can_move(x, y, 'RIGHT')
    
    # Feature 2: Distance to the nearest three coins
    coins = game_state['coins']
    distances = [(distance.euclidean((x, y), coin), coin) for coin in coins]
    distances.sort()  # Sort distances

    # Get distances to the nearest three coins, default to a large number if fewer than 3 coins
    nearest_coins = distances[:3]
    nearest_distances = [dist[0] for dist in nearest_coins] + [float('inf')] * (3 - len(distances))
    nearest_positions = [pos for _, pos in nearest_coins] + [(float('inf'), float('inf'))] * (3 - len(distances))

    # Flatten position tuples into separate features
    pos_features = [coord for pos in nearest_positions for coord in pos]


    # Combine all features into a single vector
    features = np.array(

        [int(up_possible), int(down_possible), int(left_possible), int(right_possible)] +
        nearest_distances +
        pos_features +
        [x,y],
        dtype=np.float32
    )

    #print("Feature vector:", features)
    
    return features