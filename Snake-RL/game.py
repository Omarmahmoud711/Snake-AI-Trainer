import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from queue import Queue

# Initialize Pygame
pygame.init()

# Define font for text rendering
font = pygame.font.Font("zerovelo.ttf", 25)

# Enum to represent the direction of movement
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define namedtuple for grid points
Point = namedtuple('Point', 'x, y')

# RGB colors for graphical elements
WHITE = (255, 255, 255)
RED = (200, 0, 0)
ORANGE1 = (216, 79, 42)     # Snake body color
ORANGE2 = (249, 116, 75)   # Snake border color
BLACK = (0, 0, 0)

# Size of each grid block in pixels
BLOCK_SIZE = 20

# Load images and constants
APPLE_IMAGE = pygame.image.load("apple.png")
APPLE_IMAGE = pygame.transform.scale(APPLE_IMAGE, (BLOCK_SIZE * 1.2, BLOCK_SIZE * 1.2))

BACKGROUND_IMAGE = pygame.image.load("wiese.jpg")
BACKGROUND_IMAGE = pygame.transform.scale(BACKGROUND_IMAGE, (640, 480))

SPEED = 40

# Eye properties
EYE_RADIUS = 6
EYE_COLOR = WHITE
PUPIL_RADIUS = 2
PUPIL_COLOR = BLACK

# SnakeGameAI class
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.was_trapped = False  # Track whether the snake was previously trapped
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.was_trapped = False  # Reset the trap state

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)  # Update snake's position
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        # Game over conditions
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # If the snake eats the food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # Check if the snake is trapped
        is_trapped_now = self.is_trapped()
        if is_trapped_now:
            reward -= 10  # Penalty for being trapped


        # Refresh the game screen
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_trapped(self):
        # BFS to check if there's an escape route from the snake's head to the grid's border
        visited = set()
        queue = Queue()
        queue.put(self.head)  # Start from the snake's head

        while not queue.empty():
            current = queue.get()

            if current in visited:
                continue

            visited.add(current)

            # If we've reached the grid border, there's an escape
            if current.x == 0 or current.x == self.w - BLOCK_SIZE or current.y == 0 or current.y == self.h - BLOCK_SIZE:
                return False

            # Check adjacent points for possible escape
            adjacent_points = [
                Point(current.x + BLOCK_SIZE, current.y),  # Right
                Point(current.x - BLOCK_SIZE, current.y),  # Left
                Point(current.x, current.y + BLOCK_SIZE),  # Down
                Point(current.x, current.y - BLOCK_SIZE),  # Up
            ]

            for pt in adjacent_points:
                if pt not in visited and not self.is_collision(pt):
                    queue.put(pt)

        return True  # If BFS is exhausted and we didn't find an escape, the snake is trapped

    def is_collision(self, pt=None):
        if pt is None:
            pt =self.head

        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.blit(BACKGROUND_IMAGE, (0, 0))  # Draw background

        # Determine the color based on trapped status
        snake_color = RED if self.is_trapped() else ORANGE1

        # Draw the snake using the determined color
        for pt in self.snake:
            pygame.draw.rect(self.display, snake_color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, ORANGE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        self._draw_eyes()  # Draw snake's eyes

        self.display.blit(APPLE_IMAGE, (self.food.x, self.food.y))  # Draw food
        
        # Display the score
        text = font.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])

        pygame.display.flip()  # Refresh the screen

    def _draw_eyes(self):
        if self.direction == Direction.RIGHT:
            eye1 = (self.head.x + 14, self.head.y + 6)
            eye2 = (self.head.x + 14, self.head.y + 14)
        elif self.direction == Direction.LEFT:
            eye1 = (self.head.x + 6, self.head.y + 6)
            eye2 = (self.head.x + 6, self.head.y + 14)
        elif self.direction == Direction.UP:
            eye1 = (self.head.x + 6, self.head.y + 6)
            eye2 = (self.head.x + 14, self.head.y + 6)
        elif self.direction is Direction.DOWN:
            eye1 = (self.head.x + 6, self.head.y + 14)
            eye2 = (self.head.x + 14, self.head.y + 14)

        pygame.draw.circle(self.display, EYE_COLOR, eye1, EYE_RADIUS)
        pygame.draw.circle(self.display, EYE_COLOR, eye2, EYE_RADIUS)

        pygame.draw.circle(self.display, PUPIL_COLOR, eye1, PUPIL_RADIUS)
        pygame.draw.circle(self.display, PUPIL_COLOR, eye2, PUPIL_RADIUS)

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  # Straight ahead
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Turn right
            new_dir = clock_wise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):  # Turn left
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction is Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction is Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
