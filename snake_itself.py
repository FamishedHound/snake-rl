from time import sleep

import pygame

from snake_segment import Segment


class Snake():

    def __init__(self, block_size, screen):
        self.block_size = block_size
        self.screen = screen
        self.direction = {0: (0, -1),  # up
                          1: (0, 1),  # down
                          2: (-1, 0),  # left
                          3: (1, 0)}  # right
        self.current_direction = 0
        self.snake_head = (0, 9)
        self.segments = [Segment(12, 30, self.block_size, self.screen)]

    def place_snake(self):

        # tail

        for x, y in self.snake_head[1:]:
            rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, (220, 130, 240), rect)

    def draw_snake(self):
        x, y = self.snake_head
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, (0, 250, 154), rect)

    def move(self):
        self.snake_head_x = self.snake_head[0] + self.direction[self.current_direction][0]
        self.snake_head_y = self.snake_head[1] + self.direction[self.current_direction][1]
        self.segments[0].x = self.snake_head_x
        self.segments[0].y = self.snake_head_y
        self.snake_head = (self.snake_head_x, self.snake_head_y)

        # self.move_tail()
        # self.draw_tail()

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

    def action(self, action):
        left_right_correlation = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]}

        if action == 'left':
            self.snake_head_x = self.snake_head[0] + self.direction[self.current_direction][0]
            self.snake_head_y = self.snake_head[1] + self.direction[self.current_direction][1]

        if action == 'right':
            self.snake_head_x = self.snake_head[0] + self.direction[self.current_direction][0]
            self.snake_head_y = self.snake_head[1] + self.direction[self.current_direction][1]

        if action == "up":
            self.snake_head_x = self.snake_head[0] + self.direction[self.current_direction][0]
            self.snake_head_y = self.snake_head[1] + self.direction[self.current_direction][1]

        if action == "down":
            self.snake_head_x = self.snake_head[0] + self.direction[self.current_direction][0]
            self.snake_head_y = self.snake_head[1] + self.direction[self.current_direction][1]

    def add_segment(self, x, y):

        self.segments.append(Segment(x, y, self.block_size, self.screen))
