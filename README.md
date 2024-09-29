# Bomberman RL

## Overview

This project sets out to develop a winning Reinforcement Learning (RL) agent for the classic game Bomberman. The aim is to create an AI that can effectively navigate the game environment, make strategic decisions, and maximize its score through optimal actions such as bomb placement, crate destruction, and coin collection. Our approach involves experimenting with different RL algorithms, feature spaces, and reward structures to enhance the agent's performance.

## Agent Overview

- **QL Coin:** Collects reachable coins with small rewards for proximity and large rewards for collection.
- **QL Crate:** Focuses on destroying crates by prioritizing bomb placement in dead ends, avoiding explosions, and collecting revealed coins.
- **QL Hunt:** Similar to QL Crate but maximizes crate destruction for efficient coin collection.
- **QL Chase:** Tries to target enemy players by placing bombs in positions that trap them with no escape.
- **QL Final:** Operates like QL Hunt but includes enemy proximity information and places bombs if enemies are adjacent to gain space.
- **DQN Agent:** Uses a comprehensive feature vector, including walls, bombs, enemies, and coins.
- **Small DQN Agent:** A simplified version of the DQN that uses the same feature set as QL Final, testing the effectiveness of a reduced input representation.

## Results

Our final agent prioritized survival and coin collection with an optimal bomb placement strategy in order to consistently outperform the Rule-Based Agent. Future improvements could include incorporating more information to enhance decision-making, especially further interaction with enemy players, but this would require a robust model capable of generalizing well to unknown states and significant fine-tuning.
