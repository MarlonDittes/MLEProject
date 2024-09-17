from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
FOLLOWED_INSTRUCTION = "FOLLOWED"
OPPOSITE_TO_INSTRUCTION ="OPPOSITE"

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

    # Own events:
    # compute difference between where we wanted it to move vs where it actually moved
    previous_instruction = state_to_features(old_game_state)
    moved = tuple(a - b for a,b in zip(new_game_state["self"][3], old_game_state["self"][3]))
    self.logger.info(f'Action: {ACTIONS.index(self_action)}')
    self.logger.info(f'Moved: {moved}')
    self.logger.info(f'Instruction: {previous_instruction}')

    difference = tuple(a - b for a,b in zip(previous_instruction, moved))
    if difference[0] == 0 and difference[1] == 0:
        events.append(FOLLOWED_INSTRUCTION)
    if difference[0] == 2 or difference[1] == 2:
        events.append(OPPOSITE_TO_INSTRUCTION)
        
        #if difference[0] == 0 and difference[1] == 0:
        #    events.append(FOLLOWED_INSTRUCTION)

        #if difference[0] == 2 or difference[1] == 2:
         #   events.append(OPPOSITE_TO_INSTRUCTION)


    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # update q-table (model)
    action_index = ACTIONS.index(self_action)

    best_next_action = np.argmax(self.model[state_to_features(new_game_state)])
    reward = reward_from_events(events, self.logger)
    td_target = reward + self.gamma * self.model[state_to_features(new_game_state)][best_next_action]
    td_error = td_target - self.model[state_to_features(old_game_state)][action_index]
    
    self.logger.info(self.model[state_to_features(old_game_state)])
    self.model[state_to_features(old_game_state)][action_index] += self.alpha * td_error
    self.logger.info(self.alpha * td_error)
    self.logger.info(self.model[state_to_features(old_game_state)])
    

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
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # update q-table (model)
    action_index = ACTIONS.index(last_action)

    #reward = reward_from_events(events, self.logger)
    #td_target = reward
    #td_error = td_target - self.model[state_to_features(last_game_state)][action_index]
    #self.logger.info(f'New entry for {last_action}: {self.model[state_to_features(last_game_state)][action_index]}')
    #self.model[state_to_features(last_game_state)][action_index] += self.alpha + td_error
    #self.logger.info(f'Old entry for {last_action}: {self.model[state_to_features(last_game_state)][action_index]}')

    # Store the model
    with open("model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(events: List[str], logger=None) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        FOLLOWED_INSTRUCTION: 3,
        OPPOSITE_TO_INSTRUCTION: -5,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.WAITED: -1,
        e.BOMB_DROPPED: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
