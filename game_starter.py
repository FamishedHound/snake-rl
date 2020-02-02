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
class Board():
    def __init__(self,height,width):
        self.clockobject = pygame.time.Clock()

        pygame.init()
        self.block_size=50
        self.height=height
        self.width=width
        self.screen = pygame.display.set_mode([self.block_size*self.height,self.block_size* self.width])
        self.snake = Snake(self.block_size,self.screen)


        self.running=True
        self.apple = apple(height,width,self.block_size,self.screen)

        self.collision = collision(self.apple,self.snake)
        self.apple.spawn_apple()
        self.tick=0
        self.game_manager = games_manager((self.apple.x, self.apple.y),(1,8), self.height, self.width, self.snake)
        self.current_game = self.game_manager.start_new_game(self.apple.x,self.apple.y,(1,8))
        self.epsilon = 0.1
        self.games_count=0
    def run(self):
        while self.running:

            self.clockobject.tick(900)
            self.draw_board()
            self.apple.draw_apple()
            self.tick+=1
            #self.snake.move()
            reward = self.collision.return_reward(self.height,self.width)
            self.snake.draw_snake()
            #print("reward {}".format(reward))
            print(self.tick)
            action = self.current_game.decide_action(self.epsilon,reward,self.snake.snake_head)


            self.process_ai_input(action)


            if reward==-1 or reward==1:
                self.games_count+=1

                self.snake.reset_snake()

                self.game_manager.start_new_game(self.apple.x, self.apple.y,(self.snake.snake_head))

            pygame.display.flip()

    def draw_board(self):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, (119,136,153), rect)

    def process_ai_input(self,action):
        self.snake.action(action)
        for event in pygame.event.get():

            pygame.display.update()



snake = Board(9,9)
snake.run()