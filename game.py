import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()
#random.seed(40)
# Constants
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 370  # Increased height for timer display
CELL_SIZE = 20
MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
MAZE_HEIGHT = (SCREEN_HEIGHT - 50) // CELL_SIZE  # Adjusted height for timer display
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (192, 192, 192)
BLUE = (0, 0, 255)

class Maze:
    def __init__(self, countdown_time=60, rendering_enabled=False):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) if rendering_enabled else None
        self.font = pygame.font.SysFont(None, 36)
        pygame.display.set_caption("Maze Game")
        self.clock = pygame.time.Clock()
        self.rendering_enabled=rendering_enabled
        self.maze =[[6, 4, 1, 1, 1, 1, 1, 4, 2, 4, 2, 2, 1, 2, 2, 4], 
                    [1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 4, 2, 1, 2, 1, 1, 1, 1, 4, 2, 2, 4, 2, 2, 1], 
                    [1, 2, 2, 1, 2, 4, 2, 2, 2, 1, 1, 1, 2, 4, 1, 1], 
                    [1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1], 
                    [2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 4, 1, 1], 
                    [4, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1], 
                    [2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1], 
                    [1, 1, 2, 1, 1, 2, 1, 2, 4, 2, 1, 2, 2, 2, 4, 1], 
                    [2, 1, 2, 1, 2, 2, 4, 1, 1, 2, 1, 1, 1, 1, 2, 1], 
                    [4, 1, 2, 1, 1, 4, 2, 2, 1, 2, 2, 2, 2, 4, 2, 1], 
                    [2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1], 
                    [2, 1, 4, 1, 2, 1, 2, 1, 2, 2, 2, 4, 2, 1, 4, 1], 
                    [4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1], 
                    [2, 2, 2, 1, 2, 2, 4, 2, 2, 1, 2, 1, 2, 1, 2, 1], 
                    [4, 1, 1, 1, 1, 4, 2, 2, 2, 1, 4, 2, 4, 1, 1, 3]]
        self.player_x = 0
        self.player_y = 0
        self.player_score = 0
        self.timer = Timer(countdown_time)
        self.visited = set()
        self.visited.add((self.player_y, self.player_x))  # Add the coordinates as a tuple

        self.monster_x = 8 
        self.monster_y = 9
        self.prev_mx = self.monster_x
        self.prev_my = self.monster_y
        self.monster_prev_tile = self.monster_prev_tile = self.maze[self.monster_y][self.monster_x]  # To save the previous tile where the monster

        self.no_move_count = 0  # Track the number of steps without movement
        self.max_no_move_steps = 10  # Define a threshold for no movement
        self.update_maze_with_monster()
        self.update_maze_with_player()  # 更新玩家位置
    
    def reset(self):
        self.maze =[[6, 4, 1, 1, 1, 1, 1, 4, 2, 4, 2, 2, 1, 2, 2, 4], 
                    [1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 4, 2, 1, 2, 1, 1, 1, 1, 4, 2, 2, 4, 2, 2, 1], 
                    [1, 2, 2, 1, 2, 4, 2, 2, 2, 1, 1, 1, 2, 4, 1, 1], 
                    [1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1], 
                    [2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 4, 1, 1], 
                    [4, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1], 
                    [2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1], 
                    [1, 1, 2, 1, 1, 2, 1, 2, 4, 2, 1, 2, 2, 2, 4, 1], 
                    [2, 1, 2, 1, 2, 2, 4, 1, 1, 2, 1, 1, 1, 1, 2, 1], 
                    [4, 1, 2, 1, 1, 4, 2, 2, 1, 2, 2, 2, 2, 4, 2, 1], 
                    [2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1], 
                    [2, 1, 4, 1, 2, 1, 2, 1, 2, 2, 2, 4, 2, 1, 4, 1], 
                    [4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1], 
                    [2, 2, 2, 1, 2, 2, 4, 2, 2, 1, 2, 1, 2, 1, 2, 1], 
                    [4, 1, 1, 1, 1, 4, 2, 2, 2, 1, 4, 2, 4, 1, 1, 3]]
        self.player_x = 0
        self.player_y = 0
        self.player_score = 0
        self.timer = Timer(self.timer.countdown_time)
        self.visited = set()
        self.visited.add((self.player_y, self.player_x))  # Add the coordinates as a tuple

        self.monster_x = 8 
        self.monster_y = 9
        self.prev_mx = self.monster_x
        self.prev_my = self.monster_y
        self.monster_prev_tile = self.monster_prev_tile = self.maze[self.monster_y][self.monster_x]  # To save the previous tile where the monster
        
        self.update_maze_with_monster()
        self.update_maze_with_player()  # 更新玩家位置
        return self.get_state()
    
    def get_state(self):
        return {
            'player_position': (self.player_x, self.player_y),
            'player_score': self.player_score,
            'remaining_time': self.timer.get_time(),
            'maze': self.maze
        }
    
    def render(self):
        if not self.rendering_enabled:
            return
        # Update display
        self.screen.fill(WHITE)
        self.draw_maze()
        self.timer.draw(self.screen)
        # Display the player's score
        score_text = self.font.render(f"Score: {self.player_score}", True, BLACK)
        self.screen.blit(score_text, (SCREEN_WIDTH - 150, SCREEN_HEIGHT - 40))
        pygame.display.flip()
        self.clock.tick(30)
    
    def create_maze(self):
        maze = [[0] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
        for _ in range(50):  # Randomly add obstacles
            x = random.randint(0, MAZE_WIDTH - 1)
            y = random.randint(0, MAZE_HEIGHT - 1)
            maze[y][x] = 2
        
        for _ in range(15):  # Randomly add points (incentives)
            while True:
                x = random.randint(0, MAZE_WIDTH - 1)
                y = random.randint(0, MAZE_HEIGHT - 1)
                if maze[y][x] == 1:
                    maze[y][x] = 4 
                    break
        
        maze[MAZE_HEIGHT - 1][MAZE_WIDTH - 1] = 3  
        return maze
    
    def draw_maze(self):
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                if self.maze[y][x] == 2: #Start
                    pygame.draw.rect(self.screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif self.maze[y][x] == 3: # End 
                    pygame.draw.rect(self.screen, RED, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif self.maze[y][x] == 4: # Points
                    pygame.draw.circle(self.screen, BLUE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
                elif self.maze[y][x] == 5: # Monster 
                    pygame.draw.rect(self.screen, GRAY, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))  
                elif self.maze[y][x] == 6: # Player
                    pygame.draw.rect(self.screen, GREEN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def update_maze_with_monster(self):
        # Restore the previous tile where the monster was before moving
        self.maze[self.prev_my][self.prev_mx] = self.monster_prev_tile
        # Save the current tile where the monster is moving to
        self.monster_prev_tile = self.maze[self.monster_y][self.monster_x]
        # Update the monster's position on the maze
        if self.monster_x == self.player_x and self.monster_y == self.player_y:
            return  # Handle the case where the monster collides with the player
        self.maze[self.monster_y][self.monster_x] = 5  # Assuming 4 is the value for the monster

        if self.monster_x == self.player_x and self.monster_y == self.player_y:
            return  # Handle the case where the monster collides with the player
        
        
        
        self.maze[self.monster_y][self.monster_x] = 5  # '4' represents the monster

    def update_maze_with_player(self):
        # Ensure the previous player position is marked as empty
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                if self.maze[y][x] == 6:
                    self.maze[y][x] = 1
        # Check if the player has reached the end (maze[y][x] == 3)
        if self.maze[self.player_y][self.player_x] == 3:
            return
        # Set the new player position in the maze
        self.maze[self.player_y][self.player_x] = 6
    def move_monster(self):
        # Randomly choose a direction for the monster
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            
            new_x = self.monster_x + dx
            new_y = self.monster_y + dy
            
            if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and self.maze[new_y][new_x] != 2:
                # Move the monster to the new position
                self.prev_mx = self.monster_x
                self.prev_my = self.monster_y
                self.monster_x = new_x
                self.monster_y = new_y
                break  
        # Update the maze with the new monster position
        self.update_maze_with_monster()
    def step(self, action):
        # Map action to movement
        dx, dy = 0, 0
        if action == 0:
            dx, dy = 0, -1
        elif action == 1:
            dx, dy = 0, 1
        elif action == 2:
            dx, dy = -1, 0
        elif action == 3:
            dx, dy = 1, 0
        
        new_x = self.player_x + dx
        new_y = self.player_y + dy
        reward = -0.05
        if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT:
            if self.maze[new_y][new_x] != 2:
                self.no_move_count = 0
                # Update player position
                self.player_x = new_x
                self.player_y = new_y
                
                # Add the current position to the visited set
                if (self.player_y, self.player_x) in self.visited:
                    print("Visited!")
                    reward -= 0.2  # Penalty for revisiting a position
                else:
                    print("Added")
                    self.visited.add((self.player_y, self.player_x))
                
                # Collect points
                if self.maze[self.player_y][self.player_x] == 4:
                    self.player_score += 10  # Increase score by 10 for each point
                    reward += 5
                    self.maze[self.player_y][self.player_x] = 1  # Remove the point from the maze
                if self.maze[self.player_y][self.player_x] == 3:
                    self.player_score += 100
                    reward += 100
                if self.maze[self.player_y][self.player_x] == 5:
                    self.player_score -= 500
                    reward = -10  # Penalty for losing
                    done = True  # End game when colliding with the monster
                    self.render()  # Update the display
                    return self.get_state(), reward, done  # Return immediately if collision occurs
            else:
                reward -= 0.75
                self.no_move_count += 1 
            # Update the maze with the new player position
        else:
            reward -= 0.75
            self.no_move_count += 1
        self.update_maze_with_player()
        self.move_monster()

        done = (self.maze[self.player_y][self.player_x] == 3 or
                self.monster_x == self.player_x and self.monster_y == self.player_y or
                self.timer.is_time_up())

        if self.no_move_count >= self.max_no_move_steps:
            reward = -10  # Penalty for staying in the same position
            done = True
        self.render()
        state = self.get_state()
        return state, reward, done
    def run(self):
        running = True
        won = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.step(0)
                    elif event.key == pygame.K_DOWN:
                        self.step(1)
                    elif event.key == pygame.K_LEFT:
                        self.step(2)
                    elif event.key == pygame.K_RIGHT:
                        self.step(3)
            # print(f'3:{self.no_move_count}')
            if self.no_move_count >= self.max_no_move_steps:
                won = True
                running=False

            if self.maze[self.player_y][self.player_x] == 3:
                won = True
                running=False
            if self.monster_x == self.player_x and self.monster_y == self.player_y:
                won = False
                running=False
            if self.timer.is_time_up():
                running = False
        self.end_game(won)
    
    def end_game(self, won):
        self.screen.fill(WHITE)
        if won:
            end_text = self.font.render(f'You won! Score: {self.player_score}', True, BLACK)
        else:
            end_text = self.font.render(f'Time is up! Score: {self.player_score}', True, BLACK)
        self.screen.blit(end_text, (SCREEN_WIDTH // 2 - end_text.get_width() // 2, SCREEN_HEIGHT // 2 - end_text.get_height() // 2))
        pygame.display.flip()
        pygame.time.wait(3000)
        pygame.quit()
    
    def close(self):
        pygame.quit()

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
    maze_game = Maze(countdown_time=60,rendering_enabled=True)
    maze_game.run()

if __name__ == "__main__":
    main()
