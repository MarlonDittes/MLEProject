from collections import namedtuple, deque
import pickle
from typing import List
import events as e
from .callbacks import state_to_features, ACTIONS
from .model import DQNAgent

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initializes self for training purposes, including the DQNAgent and replay memory.
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
    self.agent.save_metrics('metrics.pkl')


def reward_from_events(self, events: List[str]) -> int:
    """
    Assign rewards based on game events.
    """
    game_rewards = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -5,
        e.BOMB_DROPPED: -5,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 10,
        e.COIN_FOUND: 5,
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -20,
        e.OPPONENT_ELIMINATED: 30,
        e.SURVIVED_ROUND: 20
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
