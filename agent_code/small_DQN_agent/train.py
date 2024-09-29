from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np

# Events
TO_COIN = "TO_COIN"
OPPOSITE_TO_COIN = "OPPOSITE_TO_COIN"

TO_BOMB_POS = "TO_BOMB_POS"
OPPOSITE_TO_BOMB_POS = "OPPOSITE_TO_BOMB_POS"
GOOD_BOMB = "GOOD_BOMB"

BECAME_SAFE = "BECAME_SAFE"
BECAME_UNSAFE = "BECAME_UNSAFE"
TO_SAFETY = "TO_SAFETY"
NOT_TO_SAFETY = "NOT_TO_SAFETY"
ENSURED_DEATH = "ENSURED_DEATH"

SCARE_ENEMY = "SCARE_ENEMY"

# Actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    pass


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called after each step to capture the transition and train the model.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Convert states to feature vectors
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    # Get the index of the action
    action_index = ACTIONS.index(self_action)

    # Compute reward based on events
    reward = reward_from_events(self, events)

    # Store the transition in memory
    self.agent.remember(old_state, action_index, reward, new_state, done=False)
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each round. Perform final updates and store the model.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_state = state_to_features(last_game_state)
    action_index = ACTIONS.index(last_action)
    reward = reward_from_events(self, events)

    # Mark this as the final transition (done=True)
    self.agent.remember(last_state, action_index, reward, None, done=True)
    self.agent.replay() # train model
    self.agent.update_target_model() # update target network

    # Save the model to file
    with open("my-saved-model.pt", "wb") as file:
        self.agent.save("my-saved-model.pt")

    # Save metrics
    self.count += 1
    if self.count == 4000:
        self.agent.save_metrics('metrics.pkl')


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.WAITED: -1,
        e.BOMB_DROPPED: -1,
        e.INVALID_ACTION: -5,

        e.COIN_COLLECTED: 15,
        TO_COIN: 10,
        OPPOSITE_TO_COIN: -20,

        TO_BOMB_POS: 3,
        OPPOSITE_TO_BOMB_POS: -6,
        GOOD_BOMB: 25,

        BECAME_SAFE: 10,
        BECAME_UNSAFE: -15,
        TO_SAFETY: 10,
        NOT_TO_SAFETY: -20,
        ENSURED_DEATH: -60,

        e.GOT_KILLED: -60,
        e.KILLED_SELF: -60,
        
        SCARE_ENEMY: 50
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    return reward_sum
