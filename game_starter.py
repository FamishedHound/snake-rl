import numpy
import pygame

from apple import apple
from collision_handler import collision
from games_manager import games_manager
from snake import Snake

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
snake_starting_pos = (1,5)
class Board():
    def __init__(self,height,width):
        self.clockobject = pygame.time.Clock()

        pygame.init()
        self.block_size=50
        self.height=height
        self.width=width
        self.screen = pygame.display.set_mode([self.block_size*self.height,self.block_size* self.width])
        self.snake = Snake(self.block_size,self.screen,snake_starting_pos)


        self.running=True
        self.apple = apple(height,width,self.block_size,self.screen)

        self.collision = collision(self.apple,self.snake)
        self.apple.spawn_apple()
        self.tick=0
        self.game_manager = games_manager((self.apple.x, self.apple.y),snake_starting_pos, self.height, self.width, self.snake)
        self.current_game = self.game_manager.switch_state_table(self.apple.apple_position, self.snake.snake_head, snake_starting_pos)
        self.epsilon = 0.1
        self.games_count=0
        self.longest_streak = 0
    def run(self):
        while self.running:

            self.clockobject.tick(900)
            self.draw_board()
            self.apple.draw_apple()
            self.tick+=1

            reward = self.collision.return_reward(self.height,self.width)
            self.snake.draw_snake()
            #print(f"snake_pos before applied action{self.snake.snake_head}")
            action = self.current_game.decide_action(self.epsilon,reward,self.snake.snake_head)

            #print(f"action f{action}")
            self.process_ai_input(action)
            #print(f"snake_pos after applied action{self.snake.snake_head}")
            self.lose_win_scenario(reward)

            pygame.display.flip()

    def lose_win_scenario(self, reward):
        if reward == -1 or reward == 1:
            if reward==1:
                self.longest_streak+=1
                if self.longest_streak==110:
                    self.epsilon = 0
                print("Longest streak to eat apples without dying {}".format(self.longest_streak))

            self.games_count += 1
            #print(reward)
            if reward==-1:
                self.longest_streak=0
                self.snake.reset_snake()
            #print("new apple")
            self.apple.spawn_apple()
            self.game_manager.switch_state_table(self.apple.apple_position, self.snake.snake_head, snake_starting_pos)

    def draw_board(self):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, (119,136,153), rect)

    def process_ai_input(self,action):
        self.snake.action(action)
        for event in pygame.event.get():

            pygame.display.update()



snake = Board(6,6)
snake.run()