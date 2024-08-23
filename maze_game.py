import pygame
import random

# Initialize Pygame
pygame.init()

# Set random seed for reproducibility
random.seed(42)  # You can use any integer value for the seed

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

# Maze Class
class Maze:
    def __init__(self, countdown_time=60):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.SysFont(None, 36)
        pygame.display.set_caption("Maze Game")
        self.clock = pygame.time.Clock()

        self.maze = self.create_maze()
        self.player = Player()
        self.timer = Timer(countdown_time)

    # Create Maze with Points
    def create_maze(self):
        maze = [[0] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
        for _ in range(200):  # Randomly add obstacles
            x = random.randint(0, MAZE_WIDTH - 1)
            y = random.randint(0, MAZE_HEIGHT - 1)
            maze[y][x] = 1
        
        for _ in range(10):  # Randomly add points (incentives)
            while True:
                x = random.randint(0, MAZE_WIDTH - 1)
                y = random.randint(0, MAZE_HEIGHT - 1)
                if maze[y][x] == 0:
                    maze[y][x] = 3  # '3' represents a point
                    break
        
        maze[MAZE_HEIGHT - 1][MAZE_WIDTH - 1] = 2  # Set endpoint
        return maze

    # Draw Maze
    def draw_maze(self):
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif self.maze[y][x] == 2:
                    pygame.draw.rect(self.screen, RED, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif self.maze[y][x] == 3:
                    pygame.draw.circle(self.screen, BLUE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)

    # Main loop
    def run(self):
        running = True
        won = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.player.move(0, -1, self.maze)
                    elif event.key == pygame.K_DOWN:
                        self.player.move(0, 1, self.maze)
                    elif event.key == pygame.K_LEFT:
                        self.player.move(-1, 0, self.maze)
                    elif event.key == pygame.K_RIGHT:
                        self.player.move(1, 0, self.maze)

            self.screen.fill(WHITE)
            self.draw_maze()
            self.player.draw(self.screen)
            self.timer.draw(self.screen)

            # Display the player's score
            score_text = self.font.render(f"Score: {self.player.score}", True, BLACK)
            self.screen.blit(score_text, (SCREEN_WIDTH - 150, 600))

            if self.maze[self.player.y][self.player.x] == 2:
                won = True
                running = False

            if self.timer.is_time_up():
                running = False

            pygame.display.flip()
            self.clock.tick(30)

        self.end_game(won)
    
    # End game screen
    def end_game(self, won):
        self.screen.fill(WHITE)
        if won:
            end_text = self.font.render(f'You won! Score: {self.player.score}', True, BLACK)
        else:
            end_text = self.font.render(f'Time is up! Score: {self.player.score}', True, BLACK)
        self.screen.blit(end_text, (SCREEN_WIDTH // 2 - end_text.get_width() // 2, SCREEN_HEIGHT // 2 - end_text.get_height() // 2))
        pygame.display.flip()
        pygame.time.wait(3000)
        pygame.quit()

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
    maze_game = Maze(countdown_time=60)
    maze_game.run()

if __name__ == "__main__":
    main()