import os
import pickle
import random
import numpy as np
from .model import DQNAgent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Initializes the DQNAgent for training or playing.
    """
    # Define the input and output dimensions for the DQNAgent
    input_dim = 1734
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
        return np.zeros((16 * 16 * 6,))  # Example shape for a 16x16 field with 6 channels

    field = game_state['field'].flatten()

    # Bombs (countdown values)
    bombs = np.zeros_like(game_state['field']).flatten()
    for ((x, y), t) in game_state['bombs']:
        bombs[x + y * game_state['field'].shape[0]] = t

    # Explosion map
    explosions = game_state['explosion_map'].flatten()

    # Coins
    coins = np.zeros_like(game_state['field']).flatten()
    for (x, y) in game_state['coins']:
        coins[x + y * game_state['field'].shape[0]] = 1

    # Self position
    self_position = np.zeros_like(game_state['field']).flatten()
    self_position[game_state['self'][3][0] + game_state['self'][3][1] * game_state['field'].shape[0]] = 1

    # Other agents' positions
    others = np.zeros_like(game_state['field']).flatten()
    for other in game_state['others']:
        others[other[3][0] + other[3][1] * game_state['field'].shape[0]] = 1

    # Combine all features into a single vector
    features = np.concatenate([field, bombs, explosions, coins, self_position, others])

    return features
