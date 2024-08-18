import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 650  # Increased height for timer display
CELL_SIZE = 20
MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = (SCREEN_HEIGHT - 50) // CELL_SIZE  # Adjusted height for timer display
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (192, 192, 192)
BLUE = (0, 0, 255)

# Create Maze with Points
def create_maze():
    maze = [[0] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
    # Randomly add obstacles
    for _ in range(200):
        x = random.randint(0, MAZE_WIDTH - 1)
        y = random.randint(0, MAZE_HEIGHT - 1)
        maze[y][x] = 1
    
    # Randomly add points (incentives)
    for _ in range(10):  # Add 10 points
        while True:
            x = random.randint(0, MAZE_WIDTH - 1)
            y = random.randint(0, MAZE_HEIGHT - 1)
            if maze[y][x] == 0:  # Ensure it is not on an obstacle or endpoint
                maze[y][x] = 3  # '3' represents a point
                break
    
    # Set endpoint
    maze[MAZE_HEIGHT - 1][MAZE_WIDTH - 1] = 2
    return maze

# Draw Maze
def draw_maze(screen, maze):
    for y in range(MAZE_HEIGHT):
        for x in range(MAZE_WIDTH):
            if maze[y][x] == 1:
                pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif maze[y][x] == 2:
                pygame.draw.rect(screen, RED, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif maze[y][x] == 3:
                pygame.draw.circle(screen, BLUE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)

# Player class
class Player:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.score = 0  # Initialize the score

    def move(self, dx, dy, maze):
        new_x = self.x + dx
        new_y = self.y + dy
        if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and maze[new_y][new_x] != 1:
            self.x = new_x
            self.y = new_y
            if maze[self.y][self.x] == 3:  # Collect the point
                self.score += 10  # Increase score by 10 for each point
                maze[self.y][self.x] = 0  # Remove the point from the maze

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

# Timer class
class Timer:
    def __init__(self, countdown_time):
        self.font = pygame.font.SysFont(None, 36)
        self.start_time = pygame.time.get_ticks()
        self.countdown_time = countdown_time  # Time in seconds

    def get_time(self):
        elapsed_time = pygame.time.get_ticks() - self.start_time
        remaining_time = self.countdown_time - elapsed_time // 1000
        if remaining_time < 0:
            remaining_time = 0
        minutes = remaining_time // 60
        seconds = remaining_time % 60
        return f"Time: {minutes:02}:{seconds:02}"

    def draw(self, screen):
        time_text = self.font.render(self.get_time(), True, BLACK)
        screen.blit(time_text, (10, 600))

    def is_time_up(self):
        elapsed_time = pygame.time.get_ticks() - self.start_time
        remaining_time = self.countdown_time - elapsed_time // 1000
        return remaining_time <= 0

# Main function
def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.SysFont(None, 36)
    pygame.display.set_caption("Maze Game")
    clock = pygame.time.Clock()

    maze = create_maze()
    player = Player()
    countdown_time = 60  # Countdown time in seconds (2 minutes)
    timer = Timer(countdown_time)

    running = True
    won = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player.move(0, -1, maze)
                elif event.key == pygame.K_DOWN:
                    player.move(0, 1, maze)
                elif event.key == pygame.K_LEFT:
                    player.move(-1, 0, maze)
                elif event.key == pygame.K_RIGHT:
                    player.move(1, 0, maze)

        screen.fill(WHITE)
        draw_maze(screen, maze)
        player.draw(screen)
        timer.draw(screen)

        # Display the player's score
        score_text = font.render(f"Score: {player.score}", True, BLACK)
        screen.blit(score_text, (SCREEN_WIDTH - 150, 600))

        if maze[player.y][player.x] == 2:
            won = True
            running = False

        if timer.is_time_up():
            running = False

        pygame.display.flip()
        clock.tick(30)

    screen.fill(WHITE)
    if won:
        end_text = font.render(f'You won! Score: {player.score}', True, BLACK)
    else:
        end_text = font.render(f'Time is up! Score: {player.score}', True, BLACK)
    screen.blit(end_text, (SCREEN_WIDTH // 2 - end_text.get_width() // 2, SCREEN_HEIGHT // 2 - end_text.get_height() // 2))
    pygame.display.flip()
    pygame.time.wait(3000)

    pygame.quit()

if __name__ == "__main__":
    main()
