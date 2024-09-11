import os
import pickle
import random

import numpy as np

from random import shuffle
from collections import defaultdict


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

def value_of_bomb(location, grid, others):
    """
    This function calculates the value of a bomb at a certain location.
    :param location: The location of the bomb.
    :param grid: The current state of the game.
    :param others: The location of the other players.
    """
    value = 0
    if (location[0] == 0 or location[0] == 15) or (location[1] == 0 or location[1] == 15):
        return 0
    if grid[location[0]+1][location[1]] or grid[location[0][location[1]+1]] == 1:
        value += 1
    elif grid[location[0]-1][location[1]] or grid[location[0][location[1]-1]] == 1:
        if (value == 1):
            value += 1
        value +=1
    elif grid[location[0]+1][location[1]] or grid[location[0][location[1]+1]] == 0:
        value -= 1
    elif grid[location[0]-1][location[1]] or grid[location[0][location[1]-1]] == 0:
        if (value == -1):
            value = 0
        value -= 1
    if grid[location[0]][location[1]] == 0:
        return 0
    for other in others[3]:
        if other[0] == location[0]+1 or other[0] == location[0]-1 or other[1] == location[1]+1 or other[1] == location[1]-1:
            value += 5
        if other[0] == location[0]+2 or other[0] == location[0]-2 or other[1] == location[1]+2 or other[1] == location[1]-2:
            value += 3
        if other[0] == location[0]+3 or other[0] == location[0]-3 or other[1] == location[1]+3 or other[1] == location[1]-3:
            value += 1
    return value

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
    self.gamma = 0.1  # Discount factor
    self.epsilon = 0.1 # Exploration rate

    #RESET = True
    RESET = False


    if not os.path.isfile("my-saved-model.pt") or RESET:
        self.logger.info("Setting up model from scratch.")
        q_table = defaultdict(default_action_probabilities)
        self.model = q_table
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #print(game_state["self"][3])

    if self.train and random.uniform(0, 1) < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    features = state_to_features(game_state, self.logger)
    return ACTIONS[np.argmax(self.model[features])]

   


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
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    free_space = game_state["field"] == 0
    others_pos = [xy for (n, s, b, xy) in game_state['others']]
    for pos in others_pos:
        free_space[pos] = False
    bomb_pos = [xy for (xy, t) in game_state["bombs"]]
    for pos in bomb_pos:
        free_space[pos] = False

    d = look_for_targets(free_space, game_state['self'][3], game_state["coins"], logger)

    # This will let the agent know if there is a crate or wall in front of it
    if game_state['field'][game_state['self'][3][0]+1][game_state['self'][3][1]] == 1:
        up = 1
    if game_state['field'][game_state['self'][3][0]+1][game_state['self'][3][1]] == 0:
        up = 0
    if game_state['field'][game_state['self'][3][0]+1][game_state['self'][3][1]] == -1:
        up = -1
    if game_state['field'][game_state['self'][3][0]-1][game_state['self'][3][1]] == 1:
        down = 1
    if game_state['field'][game_state['self'][3][0]-1][game_state['self'][3][1]] == 0:
        down = 0
    if game_state['field'][game_state['self'][3][0]-1][game_state['self'][3][1]] == -1:
        down = -1
    if game_state['field'][game_state['self'][3][0]][game_state['self'][3][1]+1] == 1:
        right = 1
    if game_state['field'][game_state['self'][3][0]][game_state['self'][3][1]+1] == 0:
        right = 0
    if game_state['field'][game_state['self'][3][0]][game_state['self'][3][1]+1] == -1:
        right = -1
    if game_state['field'][game_state['self'][3][0]][game_state['self'][3][1]-1] == 1:
        left = 1
    if game_state['field'][game_state['self'][3][0]][game_state['self'][3][1]-1] == 0:
        left = 0
    if game_state['field'][game_state['self'][3][0]][game_state['self'][3][1]-1] == -1:
        left = -1
    bomb_value = value_of_bomb(game_state['self'][3], game_state['field'], game_state['others'])

    # map of bombs, work in progress
    # bomb_map = np.ones((20, 20))

    # for bomb in game_state['bombs']:
        # if bomb[1] == 3:
            # bomb_map[bomb[0]] = -3
            # bomb_map[bomb[0][0] + 1, bomb[0][1]] = -3
            # bomb_map[bomb[0][0] - 1, bomb[0][1]] = -3
            # bomb_map[bomb[0][0], bomb[0][1] + 1] = -3
            # bomb_map[bomb[0][0], bomb[0][1] - 1] = -3
            # bomb_map[bomb[0][0] + 2, bomb[0][1]] = -3
            # bomb_map[bomb[0][0] - 2, bomb[0][1]] = -3
            # bomb_map[bomb[0][0], bomb[0][1] + 2] = -3
            # bomb_map[bomb[0][0], bomb[0][1] - 2] = -3
            # bomb_map[bomb[0][0] + 3, bomb[0][1]]
        # elif bomb[1] == 2:
            # bomb_map[bomb[0]] = -2
            # bomb_map[bomb[0][0] + 1, bomb[0][1]] = -2
            # bomb_map[bomb[0][0] - 1, bomb[0][1]] = -2
            # bomb_map[bomb[0][0], bomb[0][1] + 1] = -2
            # bomb_map[bomb[0][0], bomb[0][1] - 1] = -2
            # bomb_map[bomb[0][0] + 2, bomb[0][1]] = -2
            # bomb_map[bomb[0][0] - 2, bomb[0][1]] = -2
            # bomb_map[bomb[0][0], bomb[0][1] + 2] = -2
            # bomb_map[bomb[0][0], bomb[0][1] - 2] = -2
            # bomb_map[bomb[0][0] + 3, bomb[0][1]] = -2
            # bomb_map[bomb[0][0] - 3, bomb[0][1]] = -2
            # bomb_map[bomb[0][0], bomb[0][1] + 3] = -2
            # bomb_map[bomb[0][0], bomb[0][1] - 3] = -2

        # elif bomb[1] == 1:
            # bomb_map[bomb[0]] = -1
            # bomb_map[bomb[0][0] + 1, bomb[0][1]] = -1
            # bomb_map[bomb[0][0] - 1, bomb[0][1]] = -1
            # bomb_map[bomb[0][0], bomb[0][1] + 1] = -1
            # bomb_map[bomb[0][0], bomb[0][1] - 1] = -1
            # bomb_map[bomb[0][0] + 2, bomb[0][1]] = -1
            # bomb_map[bomb[0][0] - 2, bomb[0][1]] = -1
            # bomb_map[bomb[0][0], bomb[0][1] + 2] = -1
            # bomb_map[bomb[0][0], bomb[0][1] - 2] = -1
            # bomb_map[bomb[0][0] + 3, bomb[0][1]] = -1
            # bomb_map[bomb[0][0] - 3, bomb[0][1]] = -1
            # bomb_map[bomb[0][0], bomb[0][1] + 3] = -1
            # bomb_map[bomb[0][0], bomb[0][1] - 3] = -1

        # elif bomb[1] == 0:
            # bomb_map[bomb[0]] = 0
            # bomb_map[bomb[0][0] + 1, bomb[0][1]] = 0
            # bomb_map[bomb[0][0] - 1, bomb[0][1]] = 0
            # bomb_map[bomb[0][0], bomb[0][1] + 1] = 0
            # bomb_map[bomb[0][0], bomb[0][1] - 1] = 0
            # bomb_map[bomb[0][0] + 2, bomb[0][1]] = 0
            # bomb_map[bomb[0][0] - 2, bomb[0][1]] = 0
            # bomb_map[bomb[0][0], bomb[0][1] + 2] = 0
            # bomb_map[bomb[0][0], bomb[0][1] - 2] = 0
            # bomb_map[bomb[0][0] + 3, bomb[0][1]] = 0
            # bomb_map[bomb[0][0] - 3, bomb[0][1]] = 0
            # bomb_map[bomb[0][0], bomb[0][1] + 3] = 0
            # bomb_map[bomb[0][0], bomb[0][1] - 3] = 0


    # features consist of current position, direction of nearest coin, info about surrounding tiles and value of dropping a bomb
    features = (game_state['self'][3], d, up, down, right, left, bomb_value)
    return features

def default_action_probabilities():
    weights = np.random.rand(len(ACTIONS))
    return weights / weights.sum()