from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, save_metrics

import numpy as np

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
    pass


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

    old_state = tuple(state_to_features(old_game_state))
    new_state = tuple(state_to_features(new_game_state))
    action_index = ACTIONS.index(self_action)

    # Own events:
    # compute difference between where we wanted it to move vs where it actually moved
    previous_instruction = old_state[0:2]
    moved = tuple(a - b for a,b in zip(new_game_state["self"][3], old_game_state["self"][3]))

    difference = tuple(a - b for a,b in zip(previous_instruction, moved))
    if difference[0] == 0 and difference[1] == 0:
        events.append(FOLLOWED_INSTRUCTION)
    if difference[0] == 2 or difference[1] == 2:
        events.append(OPPOSITE_TO_INSTRUCTION)


    # update q-table (model)
    reward = reward_from_events(events, self.logger)
    self.episode_rewards.append(reward)

    old_q_value = self.q_table[old_state][action_index]
    next_max_q = np.max(self.q_table[new_state])

    # Bellman update
    td_target = reward + self.gamma * next_max_q
    td_error = td_target - old_q_value
    self.episode_td_error.append(td_error)

    self.q_table[old_state][action_index] += self.alpha * td_error
    

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
    self.logger.debug(f'End of round: encountered events {", ".join(events)}')

    last_state = tuple(state_to_features(last_game_state))
    action_index = ACTIONS.index(last_action)

    # update q-table (model)
    reward = reward_from_events(events, self.logger)
    self.episode_rewards.append(reward)

    old_q_value = self.q_table[last_state][action_index]
    next_max_q = 0

    # Bellman update
    td_target = reward + self.gamma * next_max_q
    td_error = td_target - old_q_value
    self.episode_td_error.append(td_error)

    self.q_table[last_state][action_index] += self.alpha * td_error

    # Store the model
    with open("q_table.pt", "wb") as file:
        pickle.dump(self.q_table, file)

    # Append to metrics
    self.rewards.append(np.average(self.episode_rewards))
    self.td_error.append(np.average(self.episode_td_error))
    self.epsilon_history.append(self.epsilon)

    # Reset per episode
    self.episode_rewards = []
    self.episode_td_error = []
    
    self.episode_number += 1  # Increment episode count
    self.logger.info(f"Episode {self.episode_number} finished. Epsilon: {self.epsilon}")

    # Save metrics
    if self.episode_number == 300:
        save_metrics(self, 'metrics.pkl')


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
