from time import sleep

import pygame

from snake_segment import Segment


class Snake():

    def __init__(self, block_size, screen,starting_pos):
        self.block_size = block_size
        self.screen = screen
        self.starting_pos = starting_pos
        self.current_direction = 0
        self.snake_head_x = starting_pos[0]
        self.snake_head_y = starting_pos[1]
        self.snake_head = starting_pos
        self.segments = [Segment(12, 30, self.block_size, self.screen)]

    def draw_snake(self):
        x, y = self.snake_head
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        #print(" {} screen".format(self.screen))
        pygame.draw.rect(self.screen, (0, 250, 154), rect)

    def action(self, action):
        actions = {0: "left",
                   1: "right",
                   2: "up",
                   3: "down"}

        if action == 0:
            self.snake_head_x = self.snake_head[0] - 1

        if action == 1:
            self.snake_head_x = self.snake_head[0] + 1

        if action == 2:
            self.snake_head_y = self.snake_head[1] - 1

        if action == 3:
            self.snake_head_y = self.snake_head[1] + 1

        self.snake_head = (self.snake_head_x, self.snake_head_y)

    def add_segment(self, x, y):

        self.segments.append(Segment(x, y, self.block_size, self.screen))
    def reset_snake(self):
        self.snake_head_x = self.starting_pos[0]
        self.snake_head_y = self.starting_pos[1]
        self.snake_head = self.starting_pos

'''
    def draw_tail(self):
        for segment in self.segments[1:]:
            segment.draw_segment()

    def move_tail(self):

        for segment in range(len(self.segments) - 1, 0, -1):
            print(segment)
            print(len(self.segments))
            self.segments[segment].x = self.segments[segment - 1].x
            self.segments[segment].y = self.segments[segment - 1].y
            print("head {} {} seg 1 {} {}".format(self.segments[segment].x, self.segments[segment].y,
                                                  self.segments[segment - 1].x, self.segments[segment - 1].y))
'''
