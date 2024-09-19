from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np
from . import rotation

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
FOLLOWED_INSTRUCTION = "FOLLOWED"
OPPOSITE_TO_INSTRUCTION ="OPPOSITE"
FLEED_FROM_BOMB = "FLEED"
NOT_FLEED_FROM_BOMB = "NOT_FLEED"
FLEEING_FROM_BOMB = "FLEEING"
STEPPED_INTO_BOMB = "ENTERED DANGER ZONE"
BOMB_DROPPED_NEXT_TO_CRATE = "BOMB_NEXT_TO_CRATE"


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def bombNotDroppedNextToCrate(old_game_state, new_game_state):
    """
    Returns True if the agent (and only the agent) dropped a bomb next to a crate.
    
    :param old_game_state: The game state before the bomb drop.
    :param new_game_state: The game state after the bomb drop.
    :return: True if the agent dropped a bomb next to a crate, False otherwise.
    """
    
    # Extract bomb and crate information from the old and new game states
    old_bombs = [bomb[0] for bomb in old_game_state['bombs']]  # Get bomb positions from old game state
    new_bombs = [bomb[0] for bomb in new_game_state['bombs']]  # Get bomb positions from new game state
    crates = new_game_state['field']     # 2D array of the game field, 1 indicates a crate
    agent_position = new_game_state['self'][3]  # Agent's current position (x, y)

    # Check if a new bomb was added at the agent's position
    new_bomb = None
    for bomb in new_bombs:
        if bomb not in old_bombs and bomb[0] == agent_position[0] and bomb[1] == agent_position[1]:
            new_bomb = bomb
            break

    if new_bomb is None:  # No new bomb was dropped by the agent
        return False

    # Get the bomb's position (which should match agent's position)
    bomb_x, bomb_y = new_bomb

    # Check the 4 neighboring positions (up, down, left, right)
    neighbors = [(bomb_x + dx, bomb_y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

    # Check if any neighboring position has a crate (represented by 1 in the field)
    for x, y in neighbors:
        if crates[x][y] == 1:  # 1 means a crate in the field
            return True  # Bomb is next to a crate

    # No bomb was dropped next to a crate
    return False

def steppedIntoBomb(old_position, new_position, bombs):
    """
    Checks if the agent has stepped into the blast radius of any bomb.
    
    :param old_position: Tuple (x, y), the agent's position before the move.
    :param new_position: Tuple (x, y), the agent's position after the move.
    :param bombs: List of tuples ((x, y), t) where (x, y) are the bomb coordinates and t is the countdown.
    
    :return: True if the agent stepped into a bomb's blast radius, False otherwise.
    """
    for bomb, countdown in bombs:
        # Bomb explosion range is 3 tiles in each direction
        bomb_x, bomb_y = bomb
        
        # Check if the old position was in the blast radius of a bomb
        if bomb_x == old_position[0] and abs(bomb_y - old_position[1]) <= 3:
            return False
        if bomb_y == old_position[1] and abs(bomb_x - old_position[0]) <= 3:
            return False
        # Check if the new position is in the blast radius of a bomb
        if bomb_x == new_position[0] and abs(bomb_y - new_position[1]) <= 3:  # Same column
            return True
        if bomb_y == new_position[1] and abs(bomb_x - new_position[0]) <= 3:  # Same row
            return True
    
    return False

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Rotate game_state
    # if the call is redundant, it does nothing, since rotation of up-left is up-left
    self.old_game_state = rotation.rotate_game_state(old_game_state, old_game_state['self'][3])
    self.new_game_state = rotation.rotate_game_state(new_game_state, new_game_state['self'][3])

    # Own events:
    # compute difference between where we wanted it to move vs where it actually moved
    previous_instruction = state_to_features(old_game_state)[1]
    if previous_instruction != None:
        difference = tuple(a - b for a,b in zip(previous_instruction, new_game_state['self'][3]))
        
        if difference[0] == 0 and difference[1] == 0:
            events.append(FOLLOWED_INSTRUCTION)

        if difference[0] == 2 or difference[1] == 2:
            events.append(OPPOSITE_TO_INSTRUCTION)
    
    # Detect if agent is fleeing from bombs
    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]
    
    # Check for bombs in the environment
    bomb_positions = [bomb[0] for bomb in old_game_state['bombs']]
    
    if bomb_positions:
        # Find the closest bomb in the previous state
        old_distances = [np.abs(old_position[0] - bomb[0]) + np.abs(old_position[1] - bomb[1]) for bomb in bomb_positions]
        closest_bomb_distance_old = min(old_distances)
        
        # Find the closest bomb in the new state
        new_distances = [np.abs(new_position[0] - bomb[0]) + np.abs(new_position[1] - bomb[1]) for bomb in bomb_positions]
        closest_bomb_distance_new = min(new_distances)
        
        # Fleeing logic
        if closest_bomb_distance_new > closest_bomb_distance_old:
            events.append(FLEEING_FROM_BOMB)
        elif closest_bomb_distance_new < closest_bomb_distance_old:
            events.append(NOT_FLEED_FROM_BOMB)
        
        # Check if agent has fled far enough from the bomb
        if closest_bomb_distance_new > 4:
            events.append(FLEED_FROM_BOMB)
        
    if steppedIntoBomb(old_game_state['self'][3], new_game_state['self'][3], new_game_state['bombs']):
        events.append(STEPPED_INTO_BOMB)

    if bombNotDroppedNextToCrate(old_game_state, new_game_state):
        events.append(BOMB_DROPPED_NEXT_TO_CRATE)
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(events, self.logger)))

    # update q-table (model)
    #Convert the action to the unrotated frame befoer updating the q-table
    unrotated_actions = rotation.rotate_actions(old_game_state['self'][3])
    unrotated_action = unrotated_actions[self_action] # get the action in the unrotated frame
    action_index = ACTIONS.index(self_action)

    best_next_action_index = np.argmax(self.model[state_to_features(new_game_state)])
    reward = reward_from_events(events, self.logger)
    best_next_action = unrotated_actions[ACTIONS[best_next_action_index]]  # Convert to unrotated frame

    td_target = reward + self.gamma * self.model[state_to_features(new_game_state)][best_next_action_index]
    td_error = td_target - self.model[state_to_features(old_game_state)][action_index]
    self.logger.info(f'Old entry for {self_action}: {self.model[state_to_features(old_game_state)][action_index]}')
    self.model[state_to_features(old_game_state)][action_index] += self.alpha * td_error
    self.logger.info(self.alpha + td_error)
    self.logger.info(f'New entry for {self_action}: {self.model[state_to_features(old_game_state)][action_index]}')
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
        # Rotate game_state
    # if the call is redundant, it does nothing, since rotation of up-left is up-left
    self.old_game_state = rotation.rotate_game_state(last_game_state, last_game_state['self'][3])
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(events, self.logger)))
    # Convert last action to unrotated form before updating the Q-table
    unrotated_actions = rotation.rotate_actions(self.old_game_state['self'][3])
    unrotated_last_action = unrotated_actions[last_action]  # Get the unrotated action
    action_index = ACTIONS.index(unrotated_last_action)

    reward = reward_from_events(events, self.logger)
    # Append the reward to a file
    with open("rewards.csv", "a") as file:
        file.write(f"{reward}\n")
        file.write(f"END OF GAME\n")
    td_target = reward
    td_error = td_target - self.model[state_to_features(last_game_state)][action_index]
    self.logger.info(f'New entry for {last_action}: {self.model[state_to_features(last_game_state)][action_index]}')
    self.model[state_to_features(last_game_state)][action_index] += self.alpha + td_error
    self.logger.info(f'Old entry for {last_action}: {self.model[state_to_features(last_game_state)][action_index]}')

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(events: List[str], logger=None) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_DOWN: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        OPPOSITE_TO_INSTRUCTION: -5, 
        FOLLOWED_INSTRUCTION: 3,
        e.INVALID_ACTION: -10,
        e.WAITED: -2,
        e.BOMB_DROPPED: -1,
        e.CRATE_DESTROYED: 10,
        e.COIN_COLLECTED: 40,
        e.KILLED_OPPONENT: 200,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -40,
        e.OPPONENT_ELIMINATED: 50,
        e.SURVIVED_ROUND: -1,
        FLEED_FROM_BOMB: 4,
        NOT_FLEED_FROM_BOMB: -5,
        FLEEING_FROM_BOMB: 3,
        STEPPED_INTO_BOMB: -2,
        BOMB_DROPPED_NEXT_TO_CRATE: 5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
        # Append the reward to a file
    with open("rewards.csv", "a") as file:
        file.write(f"{reward_sum}\n")
    return reward_sum
