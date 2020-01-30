import pygame




class Segment():
    def __init__(self, x, y, block_size, screen):

        self.x = x
        self.y = y
        self.block_size = block_size
        self.screen = screen

        self.segment_color = (255, 99, 71)

    def draw_segment(self):
        rect = pygame.Rect(self.x * self.block_size, self.y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.segment_color, rect)