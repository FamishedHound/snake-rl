import random

import pygame


class apple():
    def __init__(self, height, width, block_size, screen, range_of_apple_spawn, snake):
        self.snake = snake
        self.height = height
        self.width = width
        self.block_size = block_size
        self.screen = screen
        self.apple_spawn_range = range_of_apple_spawn

        self.x = random.randint(0, self.width - 1)
        self.y = random.randint(0, self.width - 1)
        while self.is_apple_on_snake(self.x, self.y):
            self.x = random.randint(0, self.width - 1)
            self.y = random.randint(0, self.width - 1)
        self.apple_position = (self.x, self.y)

    def spawn_apple(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.width - 1)
        while self.is_apple_on_snake(x, y) or (x == self.x or y == self.y):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.width - 1)
        print("position of apple {} {}".format(self.x, self.y))
        #self.apple_position = (x, y) # was before the change #for moving apple____
        # self.x = x
        # self.y = y
        # for moving apple ____
        return x, y

    def move_apple(self, curr_apple_pos, target_pos, crawl_flag=True):
        cur_app_x, curr_app_y = curr_apple_pos
        dest_app_x, dest_app_y = target_pos

        if crawl_flag:
            if curr_app_y != dest_app_y:
                if curr_app_y < dest_app_y:
                    curr_app_y += 1
                else:
                    curr_app_y -= 1
            else:
                if cur_app_x < dest_app_x:
                    cur_app_x += 1
                else:
                    cur_app_x -= 1
            self.x = cur_app_x
            self.y = curr_app_y
            self.apple_position = (self.x, self.y)
            return cur_app_x, curr_app_y
        else:
            self.x = dest_app_x
            self.y = dest_app_y
            self.apple_position = (self.x, self.y)
            return dest_app_x, dest_app_y
        

    def is_apple_on_snake(self, x, y):

        for segment in self.snake.segments:
            if segment.x == x and segment.y == y:
                return True

        if self.snake.snake_head_x == x and self.snake.snake_head_y == y:
            return True
        return False

    def draw_apple(self):
        rect = pygame.Rect(self.x * self.block_size, self.y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen.get_surface(), (128, 128, 128), rect)
