import random

import pygame


class apple():
    def __init__(self, height, width, block_size, screen,range_of_apple_spawn,snake):
        self.snake = snake
        self.height = height
        self.width = width
        self.block_size = block_size
        self.screen = screen
        self.apple_spawn_range = range_of_apple_spawn
    def spawn_apple(self):
        self.x = random.randint(0,self.width-1)
        self.y = random.randint(0,self.width-1)
        while self.is_apple_on_snake(self.x,self.y):
            self.x = random.randint(0, self.width-1)
            self.y = random.randint(0, self.width-1)

        #debug

        self.apple_position = (self.x, self.y)


    def is_apple_on_snake(self,x,y):

        for segment in self.snake.segments:
            if segment.x==x and segment.y==y:
                return True

        if self.snake.snake_head_x == x and self.snake.snake_head_y == y:
            return True
        return False
    def draw_apple(self):
        rect = pygame.Rect(self.x * self.block_size, self.y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen.get_surface(), (128, 128, 128), rect)
