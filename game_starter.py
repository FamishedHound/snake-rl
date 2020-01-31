import pygame

from apple import apple
from collision_handler import collision
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

    def run(self):
        while self.running:
            self.clockobject.tick(15)
            self.draw_board()
            self.apple.draw_apple()

            #self.snake.move()
            self.collision.check_collision_apple_snake()
            self.snake.draw_snake()



            self.process_user_input()
            self.losing_condition()
            pygame.display.flip()

    def draw_board(self):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, (119,136,153), rect)

    def process_user_input(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.snake.action("left")
                if event.key == pygame.K_RIGHT:
                    self.snake.action("right")
                if event.key == pygame.K_UP:
                    self.snake.action("up")
                if event.key == pygame.K_DOWN:
                    self.snake.action("down")
    def losing_condition(self):
        if self.snake.snake_head_x<0 or self.snake.snake_head_x >self.height-1:
            exit(1)
        if self.snake.snake_head_y < 0 or self.snake.snake_head_y > self.width-1:
            exit(1)


snake = Board(9,9)
snake.run()