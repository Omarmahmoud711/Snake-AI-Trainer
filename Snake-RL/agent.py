import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point  # Import game-related classes
from model import Linear_QNet, QTrainer       # Import neural network and trainer
from helper import plot                      # Import plotting utility

# Configuration constants for the agent
MAX_MEMORY = 100_000  # Maximum memory for replay buffer
BATCH_SIZE = 1000     # Batch size for training
LR = 0.001            # Learning rate for the optimizer

# Class representing the reinforcement learning agent
class Agent:
    def __init__(self):
        self.n_games = 0  # Number of games played
        self.epsilon = 0  # Epsilon for exploration-exploitation trade-off
        self.gamma = 0.9  # Discount factor for Q-learning
        self.memory = deque(maxlen=MAX_MEMORY)  # Replay memory with maximum size
        self.model = Linear_QNet(11, 256, 3)   # Q-learning model with 11 inputs, 256 hidden units, and 3 outputs
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # QTrainer for training the model

    # Get the state of the game for the agent's decision-making
    def get_state(self, game):
        head = game.snake[0]  # Get the snake's head
        # Define points to check for collisions based on the head's position
        point_l = Point(head.x - 20, head.y)  # Left
        point_r = Point(head.x + 20, head.y)  # Right
        point_u = Point(head.x, head.y - 20)  # Up
        point_d = Point(head.x, head.y + 20)  # Down
        
        # Check the direction of movement
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Create a state representation for the agent to learn from
        state = [
            # Danger when moving straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger when turning right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger when turning left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Current movement direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location relative to the snake
            game.food.x < game.head.x,  # Food is to the left
            game.food.x > game.head.x,  # Food is to the right
            game.food.y < game.head.y,  # Food is above
            game.food.y > game.head.y  # Food is below
        ]

        return np.array(state, dtype=int)  # Return state as a numpy array

    # Add a memory to the agent's replay buffer
    def remember(self, state, action, reward, next_state, done):
        # Append the experience to the memory
        self.memory.append((state, action, reward, next_state, done))

    # Train the agent with experiences from the replay buffer (long memory)
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:  # If there's enough memory for a mini-batch
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Get a random sample
        else:
            mini_sample = self.memory  # Use all available memory if less than batch size
        
        # Separate mini-sample into components for easier handling
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)  # Train with mini-sample
    
    # Train the agent with a single step of experience (short memory)
    def train_short_memory(self, state, action, reward, next_state, done):
        # Train using the provided experience
        self.trainer.train_step(state, action, reward, next_state, done)

    # Get the agent's action based on the current state
    def get_action(self, state):
        # Determine the exploration-exploitation trade-off
        self.epsilon = 80 - self.n_games  # Decrease epsilon with more games
        final_move = [0, 0, 0]  # Default move vector

        # Random moves for exploration
        if random.randint(0, 200) < self.epsilon:  # Exploration chance based on epsilon
            move = random.randint(0, 2)  # Randomly select a move
            final_move[move] = 1  # Set the chosen move
        else:
            # Use the model to predict the best move
            state0 = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
            prediction = self.model(state0)  # Get predictions from the model
            move = torch.argmax(prediction).item()  # Select the best move
            final_move[move] = 1  # Set the chosen move

        return final_move  # Return the selected move

# Main function for training the agent
def train():
    plot_scores = []  # List to track individual game scores
    plot_mean_scores = []  # List to track mean scores over time
    total_score = 0  # Total score for mean calculation
    record = 0  # Track the highest score achieved
    agent = Agent()  # Create a new agent
    game = SnakeGameAI()  # Create a new game instance
    
    # Infinite loop to continuously train the agent
    while True:
        # Get the current state of the game
        state_old = agent.get_state(game)
        
        # Get the agent's action based on the current state
        final_move = agent.get_action(state_old)
        
        # Perform the action and get new game state
        reward, done, score = game.play_step(final_move)  # Play one step in the game
        state_new = agent.get_state(game)  # Get the new game state
        
        # Train the agent's short-term memory with the step's experience
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember the experience for future training
        agent.remember(state_old, final_move, reward, state_new, done)

        # If the game is over, handle the game-over logic
        if done:
            game.reset()  # Reset the game state
            agent.n_games += 1  # Increment the number of games played
            
            # Train the agent's long-term memory with the collected experiences
            agent.train_long_memory()
            
            # Update the record if the current score is higher
            if score > record:
                record = score
                agent.model.save()  # Save the model if record is broken
            
            # Display the game results
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Update the score tracking for plots
            plot_scores.append(score)
            total_score += score  # Add to the total score
            mean_score = total_score / agent.n_games  # Calculate mean score
            plot_mean_scores.append(mean_score)  # Add to mean score tracking
            
            # Plot the scores and mean scores
            plot(plot_scores, plot_mean_scores)

# If the script is being executed directly, start training the agent
if __name__ == '__main__':
    train()  # Begin training
