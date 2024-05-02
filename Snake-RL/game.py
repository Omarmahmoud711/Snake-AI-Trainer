import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize Pygame
pygame.init()

# Define font for text rendering
font = pygame.font.Font('arial.ttf', 25)  # Font with size 25

# Enum to represent the direction of movement
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define namedtuple for grid points
Point = namedtuple('Point', 'x, y')

# RGB colors for graphical elements
WHITE = (255, 255, 255)  # Text color
RED = (200, 0, 0)       # Food color
BLUE1 = (0, 0, 255)     # Snake body color
BLUE2 = (0, 100, 255)   # Snake border color
BLACK = (0, 0, 0)       # Background color

# Size of each grid block in pixels
BLOCK_SIZE = 20

# Game speed, controls how quickly the game updates
SPEED = 40

# Class for the snake game with AI capabilities
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w  # Width of the game area
        self.h = h  # Height of the game area
        self.display = pygame.display.set_mode((self.w, self.h))  # Create game window
        pygame.display.set_caption('Snake')  # Set window title
        self.clock = pygame.time.Clock()  # Clock for controlling game speed
        self.reset()  # Initialize or reset the game state

    # Reset game state to start a new game
    def reset(self):
        # Initial direction and snake body
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)  # Starting position of snake head
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        
        # Initialize other game state variables
        self.score = 0           # Initial score
        self.food = None         # No food at the start
        self._place_food()       # Place first food item
        self.frame_iteration = 0 # Frame count for game logic

    # Randomly place food on the game area
    def _place_food(self):
        # Calculate random x, y coordinates for food
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)  # Set food location

        # Ensure food is not placed on the snake
        if self.food in self.snake:
            self._place_food()  # Re-calculate if food collides with snake

    # Play one step in the game with a given action
    def play_step(self, action):
        self.frame_iteration += 1  # Increment frame count
        
        # Handle user inputs (check for quit event)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()  # Exit Pygame
                quit()         # Quit the script
        
        # Move the snake based on the provided action
        self._move(action)  # Update snake's head position
        self.snake.insert(0, self.head)  # Insert the new head at the beginning
        
        # Check for collisions and other game-over conditions
        reward = 0           # Default reward value
        game_over = False    # Flag for game-over condition
        # Check for collisions with boundaries or the snake's body
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # Negative reward for game over
            return reward, game_over, self.score  # Return status to agent
        
        # If snake eats food, increase score and place new food
        if self.head == self.food:
            self.score += 1  # Increment score
            reward = 10      # Positive reward for eating food
            self._place_food()  # Place new food
        else:
            self.snake.pop()  # Remove last segment of snake (to move it forward)

        # Update the game display and clock
        self._update_ui()  # Update the game's graphical elements
        self.clock.tick(SPEED)  # Control game speed

        # Return reward, game over status, and current score
        return reward, game_over, self.score

    # Check for collisions with boundaries or the snake's body
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head  # Default to the snake's head
        # Check for collisions with boundaries
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check for collisions with the snake's body (excluding the head)
        if pt in self.snake[1:]:
            return True

        return False  # No collision found

    # Update the game's graphical elements
    def _update_ui(self):
        self.display.fill(BLACK)  # Clear the screen with the background color
        
        # Draw the snake's body
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))  # Main color
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))  # Border color
        
        # Draw the food item
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Display the current score
        text = font.render("Score: " + str(self.score), True, WHITE)  # Render text for score
        self.display.blit(text, [0, 0])  # Display the score at the top left
        
        # Update the Pygame display
        pygame.display.flip()  # Refresh the screen

    # Move the snake in the appropriate direction based on the action
    def _move(self, action):
        # Possible actions: straight, right, left
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]  # Clockwise order of directions
        idx = clock_wise.index(self.direction)  # Current direction index

        # Determine the new direction based on the action taken
        if np.array_equal(action, [1, 0, 0]):  # Straight ahead
            new_dir = clock_wise[idx]  # No change in direction
        elif np.array_equal(action, [0, 1, 0]):  # Turn right
            next_idx = (idx + 1) % 4  # Get index for right turn
            new_dir = clock_wise[next_idx]  # Set new direction
        else:  # [0, 0, 1] - Turn left
            next_idx = (idx - 1) % 4  # Get index for left turn
            new_dir = clock_wise[next_idx]  # Set new direction
        
        self.direction = new_dir  # Update the direction of movement

        # Update the head's coordinates based on the new direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE  # Move right
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE  # Move left
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE  # Move down
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE  # Move up

        # Update the head's position
        self.head = Point(x, y)
