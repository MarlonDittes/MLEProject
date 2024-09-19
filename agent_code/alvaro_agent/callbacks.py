import os
import pickle
import random

import numpy as np

from random import shuffle
from collections import defaultdict


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DECAY = 20000
P_ZERO = 0.1
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
    if grid[location[0]+1][location[1]] or grid[location[0]][location[1]+1] == 1:
        value += 10
    if grid[location[0]-1][location[1]] or grid[location[0]][location[1]-1] == 1:
        if (value == 1):
            value += 10
        value +=10
    if grid[location[0]+1][location[1]] or grid[location[0]][location[1]+1] == 0:
        value -= 10
    if grid[location[0]-1][location[1]] or grid[location[0]][location[1]-1] == 0:
        if (value == -1):
            value = 0
        value -= 10
    if grid[location[0]][location[1]] == 0:
        return 0
    for other in others[3]:
        if other[0] == location[0]+1 or other[0] == location[0]-1 or other[1] == location[1]+1 or other[1] == location[1]-1:
            value += 50
        if other[0] == location[0]+2 or other[0] == location[0]-2 or other[1] == location[1]+2 or other[1] == location[1]-2:
            value += 30
        if other[0] == location[0]+3 or other[0] == location[0]-3 or other[1] == location[1]+3 or other[1] == location[1]-3:
            value += 10
    return value


def surroundingFeatures(location, grid, bombs, others, explosion, coins):
    """
    This function creates a 7x7 grid with the agent on the center recording all information.
    :param location: coordinates of agent (x, y)
    :param grid: 16x16 grid of the field containing information about walls and crates
    :param bombs: list of bombs with their timers [(x, y), t]
    :param others: list of enemies, each with their coordinates on others[i][3]
    :param explosion: 2D numpy array stating explosion information for each tile
    :param coins: list of coordinates of coins [(x, y)]
    """
    # Initialize the 7x7 surrounding information grid with zeros (free tiles)
    information = np.zeros((5, 5))
    
    # Agent's location
    x, y = location
    
    # Helper function to mark explosions in all 4 directions
    def mark_explosion(bomb_x, bomb_y, bomb_timer):
        # The bomb explodes in the 4 cardinal directions, unless blocked
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # down, up, right, left
        
        # Mark the bomb's position
        relative_bomb_x = bomb_x - x - 2 # -3 to avoid having to use abs
        relative_bomb_y = bomb_y - y - 2
        if 0 <= relative_bomb_x < 5 and 0 <= relative_bomb_y < 5:
            information[relative_bomb_x, relative_bomb_y] = -20 if bomb_timer == 1 else -3
        
        # Spread the explosion in each direction up to 3 tiles
        for dx, dy in directions:
            for step in range(1, 3):  # Up to 3 tiles
                new_x = bomb_x + dx * step
                new_y = bomb_y + dy * step
                
                # Calculate relative position in the 7x7 matrix
                relative_new_x = new_x - x + 2
                relative_new_y = new_y - y + 2
                
                # If out of bounds in 16x16 grid, stop propagation in this direction
                if new_x < 0 or new_x >= 16 or new_y < 0 or new_y >= 16:
                    break
                
                # If out of bounds in the 7x7 matrix, we don't need to record it
                if relative_new_x < 0 or relative_new_x >= 5 or relative_new_y < 0 or relative_new_y >= 5:
                    continue
                
                # If the explosion hits a wall or crate, stop further propagation
                if grid[new_x, new_y] == -1 or grid[new_x, new_y] == 1:
                    break
                
                # Mark the explosion
                if information[relative_new_x, relative_new_y]!= -4 and information[relative_new_x, relative_new_y]!=-5:
                    information[relative_new_x, relative_new_y] = -20 if bomb_timer == 1 else -30

    # Iterate over the 7x7 window centered around the agent's location
    for i in range(-1, 3):
        for j in range(-1, 3):
            # Calculate the actual coordinates on the full 16x16 grid
            grid_x = x + i
            grid_y = y + j
            
            # Determine relative position in the 7x7 matrix
            relative_x = i + 2
            relative_y = j + 2

            # If out of bounds, mark as -1 (same as wall) and continue
            if grid_x < 0 or grid_x >= 16 or grid_y < 0 or grid_y >= 16:
                information[relative_x, relative_y] = -10
                continue
            
            # First  priority: If there's a bomb explosion timer, mark -2 or -3 based on explosion
            if explosion[grid_x, grid_y] > 0:
                if explosion[grid_x, grid_y] == 1:
                    information[relative_x, relative_y] = -40  # Explosion stays for 1 turn
                else:
                    information[relative_x, relative_y] = -50  # Explosion stays for 2+ turns
                continue  # Skip further checks

            # Second priority: If there's an enemy in 'others', mark it as 2
            if any((grid_x == other[3][0] and grid_y == other[3][1]) for other in others):
                information[relative_x, relative_y] = 20
                continue  # Skip further checks for this cell since enemy takes precedence
            

            
            # Third priority: If there's a coin, mark it as 3
            if (grid_x, grid_y) in coins:
                information[relative_x, relative_y] = 30
                continue  # Skip further checks since coin info takes precedence
            
            # Fourth priority: Check the grid for crates (1), walls (-1), or free tiles (0)
            if grid[grid_x, grid_y] == 1:
                information[relative_x, relative_y] = 10  # Crate
            elif grid[grid_x, grid_y] == -1:
                information[relative_x, relative_y] = -10  # Stone wall (not breakable)
            else:
                information[relative_x, relative_y] = 0  # Free tile
    
    # After processing the grid, now we mark bomb explosions in the 7x7 vision area
    for bomb in bombs:
        bomb_x, bomb_y, bomb_timer = bomb[0][0], bomb[0][1], bomb[1]
        # If the bomb is within the agent's 7x7 vision area, apply its explosion pattern
        if abs(bomb_x - x) <= 2 and abs(bomb_y - y) <= 2:
            mark_explosion(bomb_x, bomb_y, bomb_timer)
    
    return information


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
    self.epsilon = 0.1 # Exploration rate

    #RESET = True
    RESET = False


    if not os.path.isfile("my-saved-model.pt"):
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
    self.epsilon = float ( DECAY/ (DECAY + game_state["round"])) * P_ZERO
    if self.train and random.uniform(0, 1) < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .224, .001, .1])

    self.logger.debug("Querying model for action.")
    features = state_to_features(game_state, self.logger)
    self.logger.debug(ACTIONS[np.argmax(self.model[features])])
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
    information7x7 = surroundingFeatures(game_state['self'][3], game_state['field'], game_state['bombs'], game_state['others'], game_state['explosion_map'], game_state['coins'])
    # features consist of current position, direction of nearest coin, info about surrounding tiles and value of dropping a bomb
    features = (
        game_state['self'][3], d, up, down, right, left, bomb_value,
        information7x7[0][0], information7x7[0][1], information7x7[0][2], information7x7[0][3],
        information7x7[1][0], information7x7[1][1], information7x7[1][2], information7x7[1][3],
        information7x7[2][0], information7x7[2][1], information7x7[2][2],  information7x7[2][3]
       
    )
    return features

def default_action_probabilities():
    weights = np.random.rand(len(ACTIONS))
    return weights / weights.sum()