import pygame




class Segment():
    def __init__(self, x, y, block_size, screen,direction):
        self.direction = direction
        self.x =  x
        self.pastx = 0
        self.pasty = 0
        self.y = y
        if self.direction == 0 :
            self.x = x+1
            self.y = y
        if self.direction == 1:
            self.x = x - 1
            self.y = y
        if self.direction == 2:
            self.x = x
            self.y = y + 1
        if self.direction == 3:
            self.x = x
            self.y = y - 1
        #print(self.x,self.y)
        self.block_size = block_size
        self.screen = screen

        self.segment_color = (255, 99, 71)

    def draw_segment(self):
        rect = pygame.Rect(self.x * self.block_size, self.y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.segment_color, rect)