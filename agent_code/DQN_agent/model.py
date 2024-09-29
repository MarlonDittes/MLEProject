import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Define a simple feed-forward neural network
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 128)  # Second hidden layer
        self.fc3 = nn.Linear(128, output_dim)  # Output layer (Q-values for each action)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Return Q-values for all actions

class DQNAgent:
    def __init__(self, state_size, action_size, alpha=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99945):
        self.state_size = state_size  # Size of the input feature vector
        self.action_size = action_size  # Number of actions
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(state_size, action_size)  # The Q-network
        self.target_model = DQN(state_size, action_size)  # Target network
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.memory = deque(maxlen=2000)  # Replay memory to store transitions
        self.batch_size = 64

        # Initialize lists for plotting
        self.losses = []
        self.rewards = []
        self.epsilon_history = []

    def update_target_model(self):
        """Copy weights from the main model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store a transition in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        else:
            state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
            q_values = self.model(state).detach().numpy()
            return np.argmax(q_values)  # Exploit: choose action with the highest Q-value

    def replay(self):
        """Train the model by sampling a batch of experiences from the replay memory."""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples yet to train

        minibatch = random.sample(self.memory, self.batch_size)
        batch_rewards = []  # To store rewards for the current batch
        batch_losses = []

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)

            if next_state is None or done:
                target = reward  # There’s no future reward for terminal states
            else:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            target_f = self.model(state).detach()  # Get the predicted Q-values
            target_f[0][action] = target # Update the Q-value for the taken action

            # Perform a gradient descent step
            self.optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

            # Record the loss for plotting
            #self.losses.append(loss.item())
            batch_losses.append(loss.item())
            batch_rewards.append(reward)

        # Calculate and store average reward for this replay batch
        avg_reward = np.mean(batch_rewards)
        self.rewards.append(avg_reward)
        avg_loss = np.mean(batch_losses)
        self.losses.append(avg_loss)

        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.epsilon_history.append(self.epsilon)

    def save(self, file_name):
        """Save the model parameters."""
        torch.save(self.model.state_dict(), file_name)

    def load(self, file_name):
        """Load the model parameters."""
        self.model.load_state_dict(torch.load(file_name))
        self.update_target_model()  # Update the target network as well

    def save_metrics(self, filename='metrics.pkl'):
        """Save the metrics to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'losses': self.losses,
                'rewards': self.rewards,
                'epsilon_history': self.epsilon_history
            }, f)
