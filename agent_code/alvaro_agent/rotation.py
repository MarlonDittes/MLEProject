import numpy as np

def rotate_actions(position):
    x, y = position
    if x <= 8 and y <= 8:  # Upper-left quadrant
        actions = {'UP': 'UP', 'RIGHT': 'RIGHT', 'DOWN': 'DOWN', 'LEFT': 'LEFT', 'BOMB': 'BOMB', 'WAIT': 'WAIT'}
    elif x <= 8 and y > 8:  # Upper-right quadrant
        actions = {'UP': 'LEFT', 'RIGHT': 'UP', 'DOWN': 'RIGHT', 'LEFT': 'DOWN', 'BOMB': 'BOMB', 'WAIT': 'WAIT'}
    elif x > 8 and y <= 8:  # Lower-left quadrant
        actions = {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP', 'BOMB': 'BOMB', 'WAIT': 'WAIT'}
    elif x > 8 and y > 8:  # Lower-right quadrant
        actions = {'UP': 'DOWN', 'RIGHT': 'LEFT', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'BOMB': 'BOMB', 'WAIT': 'WAIT'}
    return actions

def rotate_grid(grid, position):
    x, y = position
    if x <= 8 and y <= 8:  # Upper-left quadrant
        return grid
    elif x <= 8 and y > 8:  # Upper-right quadrant
        return np.rot90(grid, 1)
    elif x > 8 and y <= 8:  # Lower-left quadrant
        return np.rot90(grid, 3)
    elif x > 8 and y > 8:  # Lower-right quadrant
        return np.rot90(grid, 2)

# Helper function to rotate positions based on the current quadrant
def rotate_position(position, quadrant):
    x, y = position
    if quadrant == 'upper-right':
        return y, 16 - x  # Rotating 90 degrees clockwise
    elif quadrant == 'lower-left':
        return 16 - y, x  # Rotating 270 degrees clockwise (90 degrees counterclockwise)
    elif quadrant == 'lower-right':
        return 16 - x, 16 - y  # Rotating 180 degrees
    return position  # No rotation needed for upper-left

def get_quadrant(position):
    x, y = position
    if x <= 8 and y <= 8:
        return 'upper-left'
    elif x <= 8 and y > 8:
        return 'upper-right'
    elif x > 8 and y <= 8:
        return 'lower-left'
    elif x > 8 and y > 8:
        return 'lower-right'

# Rotates the entire game state to the upper-left quadrant
def rotate_game_state(game_state, position):
    quadrant = get_quadrant(position)  # Determine which quadrant the agent is in

    # Rotate the field grid
    rotated_grid = rotate_grid(game_state['field'], position)
    
    # Rotate the bombs' positions
    rotated_bombs = [((rotate_position(bomb[0], quadrant)), bomb[1]) for bomb in game_state['bombs']]
    
    # Rotate the coins' positions
    rotated_coins = [rotate_position(coin, quadrant) for coin in game_state['coins']]

    # Rotate the agent's position (self)
    agent_name, agent_score, can_bomb, agent_pos = game_state['self']
    rotated_agent_pos = rotate_position(agent_pos, quadrant)
    rotated_self = (agent_name, agent_score, can_bomb, rotated_agent_pos)

    # Rotate the opponents' positions (others)
    rotated_others = [
        (name, score, can_bomb, rotate_position(pos, quadrant))
        for name, score, can_bomb, pos in game_state['others']
    ]

    # Rotate explosion map
    rotated_explosion_map = rotate_grid(game_state['explosion_map'], position)

    # Build the rotated game state
    rotated_game_state = {
        'round': game_state['round'],
        'step': game_state['step'],
        'field': rotated_grid,
        'bombs': rotated_bombs,
        'explosion_map': rotated_explosion_map,
        'coins': rotated_coins,
        'self': rotated_self,
        'others': rotated_others
    }

    return rotated_game_state
