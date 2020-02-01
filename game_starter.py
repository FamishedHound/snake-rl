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

        self.game_manager = games_manager(self.height, self.width, self.snake)
        self.running=True
        self.apple = apple(height,width,self.block_size,self.screen)
        self.collision = collision(self.apple,self.snake)
        self.apple.spawn_apple()

        #self.current_game = self.game_manager.start_new_game()

    def run(self):
        while self.running:
            self.clockobject.tick(15)
            self.draw_board()
            self.apple.draw_apple()

            #self.snake.move()
            self.collision.return_reward(self.height,self.width)
            self.snake.draw_snake()



            #self.process_ai_input()

            pygame.display.flip()

    def draw_board(self):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, (119,136,153), rect)

    def process_ai_input(self,action):
        self.snake.action(action)



snake = Board(9,9)
snake.run()