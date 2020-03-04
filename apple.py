import random

import pygame


class apple():
    def __init__(self, height, width, block_size, screen,range_of_apple_spawn):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.screen = screen
        self.apple_spawn_range = range_of_apple_spawn
    def spawn_apple(self):
        self.x = random.randint(self.apple_spawn_range[0],self.apple_spawn_range[1])
        self.y = random.randint(self.apple_spawn_range[0],self.apple_spawn_range[1])


        self.apple_position = (self.x, self.y)

    def draw_apple(self):
        rect = pygame.Rect(self.x * self.block_size, self.y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen.get_surface(), (255, 255, 255), rect)
