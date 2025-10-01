# Pong with a Dueling Deep Q-Network

This project implements a Reinforcement Learning agent that learns to play the Atari game Pong. The agent is built using a Dueling Deep Q-Network (Dueling DQN) architecture with PyTorch and Gymnasium.

## Demo

Here is the trained agent playing a full game of Pong.
![pong-agent-playing-game.mp4](https://github.com/user-attachments/assets/36072835-a0ba-4a15-8ad4-c346ac17e5cd)

## Overview

The goal of Reinforcement Learning (RL) is to train an **agent** to make optimal decisions within an **environment**. The agent learns by taking **actions**, observing the resulting **state** and **reward**, and adjusting its strategy to maximize its cumulative reward over time.

In this project:
- **Agent**: A Dueling DQN model.
- **Environment**: The Atari Pong game from the `gymnasium` library.
- **State**: A stack of 4 consecutive, preprocessed game frames (84x84 pixels). Stacking frames allows the agent to infer motion, like the direction of the ball.
- **Actions**: The possible moves the agent can make (e.g., move paddle up, move paddle down).
- **Reward**: +1 for scoring a point, -1 for letting the opponent score.

## Core Concepts: The "Why" and "How"

This model is built on several key RL concepts that work together to enable stable and efficient learning.

### 1. Deep Q-Networks (DQN)

**How?** A standard Q-learning algorithm uses a table (Q-table) to store the expected future reward (Q-value) for every possible state-action pair. However, in environments with a vast number of states like Pong (pixel combinations), a Q-table is infeasibly large. A DQN solves this by using a deep neural network to *approximate* the Q-function: `Q(state, action) ≈ reward`.

**Why?** The neural network can generalize from states it has seen to new, similar states. Instead of memorizing every single screen, it learns to recognize important features like the ball's position, the paddle's position, and their velocities.

### 2. Dueling DQN Architecture

**How?** A standard DQN outputs a Q-value for each action directly. A Dueling DQN splits this into two separate streams (or "heads") within the network:
1.  **Value Stream (V(s))**: Estimates how valuable it is to be in a given state `s`. It answers, "How good is my current situation?"
2.  **Advantage Stream (A(s, a))**: Estimates how much better a given action `a` is compared to all other possible actions in that state. It answers, "How much better is moving up versus moving down right now?"

These two streams are then combined to produce the final Q-values.

**Why?** This separation is more efficient. The network can learn the value of a state (e.g., a state where the ball is about to fly past the opponent is very valuable) without having to learn the Q-value for every single action in that state. This leads to better policy evaluation and faster learning, especially in action-rich environments.

### 3. Experience Replay

**How?** As the agent plays, it stores its experiences—the `(state, action, reward, next_state, done)` tuples—in a large memory buffer. During training, instead of learning from the most recent experience, the agent samples a random mini-batch of experiences from this buffer.

**Why?** This technique has two main benefits:
1.  **Breaks Correlations**: Consecutive experiences in a game are highly correlated. Training on random samples breaks these correlations, stabilizing the learning process and preventing the model from getting stuck in feedback loops.
2.  **Data Efficiency**: Each experience can be reused multiple times for training, making the learning process more efficient.

### 4. Target Network

**How?** Two neural networks are used: a **main network** that is constantly being updated, and a **target network** whose weights are frozen. The main network's Q-values are updated towards a target Q-value calculated by the target network. Periodically (e.g., every 10,000 frames), the weights from the main network are copied over to the target network.

**Why?** This creates a stable learning target. If we were to use the same network for both predicting Q-values and calculating the target, the target would shift with every training step. This is like trying to hit a moving target. By keeping the target network fixed for a period, we provide a stable objective, which prevents oscillations and helps the training converge.

## Project Structure

```
├── checkpoints/
│   └── checkpoint_ep5000_...pth
├── videos/
│   └── pong-agent-episode-0.mp4
├── training.ipynb
├── record_video.py
├── logs_statistics.ipynb
└── README.md
```

- **`training.ipynb`**: The main script to train the agent from scratch or resume from a checkpoint.
- **`record_video.py`**: A script to generate an `.mp4` video of the agent playing one episode.
- **`checkpoints/`**: Directory where model weights (`.pth` files) are saved during training.
- **`videos/`**: Directory where recorded gameplay videos are saved.
- **`logs_statistics.ipynb`**: Extened statistics script to analyze training logs
- **`plots.ipynb`**: Plots I used for my uni report
