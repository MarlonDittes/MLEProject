from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, save_metrics

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
    agent_movement = tuple(a - b for a,b in zip(new_game_state["self"][3], old_game_state["self"][3]))
    # went towards coin?
    to_coin = old_state[0:2]
    if not(to_coin[0] == -1 and to_coin[1] == -1):
        coin_following = tuple(a - b for a,b in zip(to_coin, agent_movement))
        if coin_following[0] == 0 and coin_following[1] == 0:
            events.append(TO_COIN)
        if coin_following[0] == 2 or coin_following[1] == 2:
            events.append(OPPOSITE_TO_COIN)

    # went towards good bomb position
    to_bomb_pos = old_state[2:4]
    if not((to_bomb_pos[0] == -1 and to_bomb_pos[1] == -1) or (to_bomb_pos[0] == 0 and to_bomb_pos[1] == 0)):
        bomb_pos_following = tuple(a - b for a,b in zip(to_bomb_pos, agent_movement))
        if bomb_pos_following[0] == 0 and bomb_pos_following[1] == 0:
            events.append(TO_BOMB_POS)
        if bomb_pos_following[0] == 2 or bomb_pos_following[1] == 2:
            if old_state[6]:    #punish not going there if we could place a bomb
                events.append(OPPOSITE_TO_BOMB_POS)

    if to_bomb_pos[0] == 0 and to_bomb_pos[1] == 0:
        if old_state[6] and not new_state[6]:
            events.append(GOOD_BOMB)

    # went to safety?
    if old_state[4] == 0 and old_state[5] == 0:
        if not (new_state[4] == 0 and new_state[5] == 0):
            #if not(old_state[6] and not new_state[6]):   #don't punish in case of dropping a bomb
            events.append(BECAME_UNSAFE)
    if new_state[4] == 0 and new_state[5] == 0:
        if not (old_state[4] == 0 and old_state[5] == 0):
            events.append(BECAME_SAFE)

    to_safety = old_state[4:6]
    if not(to_safety[0] == 0 and to_safety[1] == 0):
        safety_following = tuple(a - b for a,b in zip(to_safety, agent_movement))
        if safety_following[0] == 0 and safety_following[1] == 0:
            events.append(TO_SAFETY)
        if safety_following[0] == 2 or safety_following[1] == 2:
            events.append(NOT_TO_SAFETY)

    new_to_safety = new_state[4:6]
    if not(to_safety[0] == -1 and to_safety[1] == -1):
        if new_to_safety[0] == -1 and new_to_safety[1] == -1:
            events.append(ENSURED_DEATH)


      # scared off enemy with bomb?
    if old_state[6] and not new_state[6]:
        if old_state[11]:
            events.append(SCARE_ENEMY)

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
    with open("rewardsFinalDQN.csv", "a") as file:
        file.write(f"{reward}\n")
        file.write(f"END OF GAME\n")
    with open("winnerFinalDQN.csv", "a") as file:
        if e.SURVIVED_ROUND in events:
            file.write(f"1\n")
        else:
            file.write(f"0\n")
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
    if self.episode_number == self.episodes:
        save_metrics(self, 'metrics.pkl')


def reward_from_events(events: List[str], logger=None) -> int:
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
    logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    with open("rewardsFinalDQN.csv", "a") as file:
        file.write(f"{reward_sum}\n")

    return reward_sum
