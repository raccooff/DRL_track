import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random

import matplotlib.pyplot as plt

import os
from collections import deque

from environment import TetrisEnv
from model import DQN

EPISODES = 2000
BATCH_SIZE = 512
GAMMA = 0.99
LR = 1e-3
MEMORY_SIZE = 30000
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_metrics_and_plots(scores, lines, epsilons, filename_prefix="training"):

    """Saves training data to CSV and generates a plot PNG."""
    
    # 1. Save data to CSV
    csv_filename = f"{filename_prefix}_metrics.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Score", "Lines_Cleared", "Epsilon"])
        for i in range(len(scores)):
            writer.writerow([i + 1, scores[i], lines[i], epsilons[i]])

    # 2. Generate and save Plots
    plt.figure(figsize=(15, 5))

    # Plot 1: Scores
    plt.subplot(1, 3, 1)
    plt.plot(scores, color='blue', alpha=0.3, label="Score")
    if len(scores) >= 50:
        # Calculate 50-episode moving average
        moving_avg = np.convolve(scores, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(scores)), moving_avg, color='blue', label="50-ep SMA")
    plt.title("Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()

    # Plot 2: Lines Cleared
    plt.subplot(1, 3, 2)
    plt.plot(lines, color='green', alpha=0.5, label="Lines Cleared")
    if len(lines) >= 50:
        lines_avg = np.convolve(lines, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(lines)), lines_avg, color='darkgreen', label="50-ep SMA")
    plt.title("Lines Cleared per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Lines")
    plt.legend()

    # Plot 3: Epsilon Decay
    plt.subplot(1, 3, 3)
    plt.plot(epsilons, color='orange', label="Epsilon")
    plt.title("Exploration Rate (Epsilon)")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.tight_layout()
    plot_filename = f"{filename_prefix}_plot.png"
    plt.savefig(plot_filename)
    plt.close()


def train():

    env = TetrisEnv()
    model = DQN(env.state_size).to(device)

    target_model = DQN(env.state_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # mse for regression of q-values

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START

    # metrics
    scores_history = []
    lines_history = []
    epsilon_history = []

    for episode in range(1, EPISODES + 1):
        
        env.reset()
        env.spawn_piece()
        done = False
        
        while not done:

            actions = env.get_possible_actions()

            if not actions:
                break
                
            # evaluate the states for the current piece
            next_states = []
            valid_actions = []

            for action in actions:

                state = env.get_state_for_action(action)

                if state is not None:
                    next_states.append(state)
                    valid_actions.append(action)
                    
            if not valid_actions:
                break
                
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
            
            # epsilon-greedy action selection
            # with probability epsilon select a random action, otherwise select the action with the highest Q-value
            if random.random() < epsilon:
                best_idx = random.randint(0, len(valid_actions) - 1)
            else:
                with torch.no_grad():

                    q_values = model(next_states_tensor).squeeze(1)
                    best_idx = torch.argmax(q_values).item()

            best_action = valid_actions[best_idx]
            chosen_state = next_states[best_idx]

            _, reward, done, info = env.step(best_action)
            
            # see next piece's possible states for target Q-value
            next_next_states = []
            if not done:
                next_actions = env.get_possible_actions()

                for a in next_actions:
                    s = env.get_state_for_action(a)
                    if s is not None:
                        next_next_states.append(s)

            # store transition in memory
            memory.append((chosen_state, reward, done, next_next_states))

            # learn from memory if we have enough samples

            if len(memory) >= BATCH_SIZE:

                batch = random.sample(memory, BATCH_SIZE)
                
                b_states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32, device=device)
                b_rewards = torch.tensor([t[1] for t in batch], dtype=torch.float32, device=device)
                b_dones = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=device)
                
                # q targets
                b_targets = b_rewards.clone()

                for i in range(BATCH_SIZE):

                    if not b_dones[i] and len(batch[i][3]) > 0:

                        nn_states = torch.tensor(np.array(batch[i][3]), dtype=torch.float32, device=device) # next next states for target q value

                        with torch.no_grad():
                            max_q_next = target_model( nn_states ).max().item() # max q value
                            b_targets[i] += GAMMA * max_q_next # q target = reward + gamma * (max q value of next states)
                            
                b_targets = b_targets.unsqueeze(1)
                
                # current q values
                current_q = model(b_states)
                
                loss = criterion(current_q, b_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update metrics
        scores_history.append(info['score'])
        lines_history.append(info['lines_cleared'])
        epsilon_history.append(epsilon)

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 10 == 0:
            print(f"Episode: {episode:4} | Score: {info['score']:6.1f} | Lines: {info['lines_cleared']:4} | Epsilon: {epsilon:.3f}")

        # Save model and update plots periodically
        if episode % 100 == 0:
            torch.save(model.state_dict(), "tetris_dqn.pt")
            save_metrics_and_plots(scores_history, lines_history, epsilon_history)

    # Final save at the end of training
    torch.save(model.state_dict(), "tetris_dqn.pt")
    save_metrics_and_plots(scores_history, lines_history, epsilon_history)
    print("Training Complete. Model saved to 'tetris_dqn.pt'. Data saved to 'training_metrics.csv' and 'training_plot.png'.")

if __name__ == "__main__":

    train()