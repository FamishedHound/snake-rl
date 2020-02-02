import random

import pygame
class apple():
    def __init__(self,height,width,block_size,screen):

        self.height = height
        self.width=width
        self.block_size = block_size
        self.screen = screen



    def spawn_apple(self):
        self.x = 7
        self.y = 4

        self.apple_position = (self.x,self.y)

    def draw_apple(self):
        rect = pygame.Rect(self.x * self.block_size, self.y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, (139,0,0), rect)




