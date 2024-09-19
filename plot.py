import matplotlib.pyplot as plt

import numpy as np

def plot_rewards(file_path, title, bin_size=200):
    """
    Reads the rewards from the file and plots the average rewards per bin (group of rounds) using scatter plot.
    Ignores "END OF GAME" lines.
    
    :param file_path: Path to the file containing the rewards.
    :param bin_size: The number of rounds to group together in each bin.
    """
    rewards = []

    # Read rewards from the file
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line == "END OF GAME" or not line:  # Skip "END OF GAME" and empty lines
                continue
            try:
                rewards.append(int(line))  # Add valid reward lines to the list
            except ValueError:
                print(f"Invalid line in rewards file: {line}")  # Handle any unexpected data

    # Bin the rewards every `bin_size` rounds
    if rewards:
        binned_rewards = [sum(rewards[i:i + bin_size]) / bin_size for i in range(0, len(rewards), bin_size)]
        
        # Use scatter plot for the binned rewards
        plt.scatter(range(len(binned_rewards)), binned_rewards)
        plt.xlabel(f'Binned Rounds (every {bin_size} rounds)')
        plt.ylabel('Average Reward')
        plt.title(f'Average Rewards per {bin_size} Rounds, {title}')
        plt.show()
    else:
        print("No rewards data available to plot.")

def plot_rewards_by_game(file_path, title):
    """
    Reads the rewards from the file and bins them by games, then plots the total rewards per game using scatter plot.
    It sums all the rewards between "END OF GAME" markers.
    
    :param file_path: Path to the file containing the rewards.
    """
    game_rewards = []
    current_game_reward = 0

    # Read rewards from the file and bin by game
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line == "END OF GAME":  # End of a game
                game_rewards.append(current_game_reward)
                current_game_reward = 0  # Reset for the next game
            elif line:  # If it's a reward line, add it to the current game
                try:
                    current_game_reward += int(line)  # Sum rewards for the current game
                except ValueError:
                    print(f"Invalid line in rewards file: {line}")  # Handle any unexpected data

    # Use scatter plot for the total rewards per game
    if game_rewards:
        plt.scatter(range(len(game_rewards)), game_rewards)
        plt.xlabel('Game')
        plt.ylabel('Total Reward')
        plt.title(f'Total Rewards per Game, {title}')
        plt.show()
    else:
        print("No game rewards data available to plot.")

def plot_rewards_by_game_fit(file_path, title):
    """
    Reads the rewards from the file and bins them by games, then plots the total rewards per game using scatter plot.
    Fits a curve to the data and plots the 1-sigma bounds.
    
    :param file_path: Path to the file containing the rewards.
    """
    game_rewards = []
    current_game_reward = 0

    # Read rewards from the file and bin by game
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line == "END OF GAME":  # End of a game
                game_rewards.append(current_game_reward)
                current_game_reward = 0  # Reset for the next game
            elif line:  # If it's a reward line, add it to the current game
                try:
                    current_game_reward += int(line)  # Sum rewards for the current game
                except ValueError:
                    print(f"Invalid line in rewards file: {line}")  # Handle any unexpected data

    if game_rewards:
        # Scatter plot for the total rewards per game
        games = np.arange(len(game_rewards))
        plt.scatter(games, game_rewards, color='blue', label='Total Rewards per Game')

        # Linear fit for the total rewards per game
        coeffs = np.polyfit(games, game_rewards, deg=1)
        poly = np.poly1d(coeffs)

        # Generate fitted line data
        fit_values = poly(games)

        # Calculate 1-sigma (standard deviation) of the residuals
        residuals = game_rewards - fit_values
        sigma = np.std(residuals)

        # Plot the fit line
        plt.plot(games, fit_values, color='red', label='Fit Line', linestyle='--')

        # Plot horizontal line at y = 0
        plt.axhline(0, color='black', linestyle='-', linewidth=10, label='y = 0')
        
        # Plot 1-sigma bounds as shaded regions
        plt.fill_between(games, fit_values - sigma, fit_values + sigma, color='red', alpha=0.2, label='1-Sigma Bounds')

        # Add labels and title
        plt.xlabel('Game')
        plt.ylabel('Total Reward')
        plt.title(f'Total Rewards per Game, {title}')
        
        # Add legend
        plt.legend()

        # Show plot
        plt.show()

    else:
        print("No game rewards data available to plot.")

# Example usage
plot_rewards_by_game_fit("agent_code/alvaro_agent/rewardsDefaultAlone.txt", "Default x3, tweaked training")

#plot_rewards_by_game_fit("agent_code/alvaro_agent/tweakedRewardsCoinHeavenx3.txt", "Coin Heaven x3, tweaked training")

#plot_rewards_by_game_fit("agent_code/alvaro_agent/rewardsDefaultAlone.txt", "Default Alone")
#plot_rewards_by_game_fit("agent_code/alvaro_agent/rewardsCoinHeaven.txt", "Coin Heaven")
#plot_rewards_by_game_fit("agent_code/alvaro_agent/rewardsCoinHeavenx3.txt", "Coin Heaven x3")
#plot_rewards_by_game_fit("agent_code/alvaro_agent/oldRewardsCoinHeavenx3.txt", "Coin Heaven, Old x3")

