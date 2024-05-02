# Reinforcement Learning Snake Game AI

This project implements a reinforcement learning (RL) agent that learns to play the classic Snake game using a neural network and Q-learning. The project consists of three main components:

1. **Game**: The implementation of the Snake game with basic logic, rendering, and user input handling.
2. **Model**: The neural network architecture and the training logic for the RL agent.
3. **Agent**: The RL agent that interacts with the game, trains the model, and learns from its experience.



## Project Structure

- `game.py`: Contains the `SnakeGameAI` class, which implements the Snake game logic, including rendering, game rules, and user input.
- `model.py`: Contains the `Linear_QNet` class, a simple neural network architecture for Q-learning, and the `QTrainer` class, which handles training and optimization.
- `agent.py`: Contains the `Agent` class, which implements the RL agent, memory management, and action selection.
- `helper.py`: Contains utility functions for plotting training results, used to visualize scores during training.
- `README.md`: This file, providing an overview of the project.

## Getting Started

To run this project on your local environment, ensure you have Python installed, along with the necessary dependencies like `PyTorch`, `Pygame`, and `Matplotlib`. You can set up a virtual environment to manage dependencies and keep your environment clean.
