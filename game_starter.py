import pickle

import numpy as np
import pygame
import matplotlib.pyplot as plt
from DQN.DQN_agent import DQN_agent
from Function_approx.approximator import f_approximation
from apple import apple
from collision_handler import collision
from games_manager import games_manager
from snake import Snake
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

snake_starting_pos = (1, 5)
# ( bool (do we want to load) ,  filename )
load_tables_from_file = (False, "9x9model")

range_of_apple_spawn = (1, 4)


class Board():
    def __init__(self, height, width):
        self.clockobject = pygame.time.Clock()

        pygame.init()
        self.block_size = 50
        self.height = height
        self.width = width
        self.screen = pygame.display.set_mode([self.block_size * self.height, self.block_size * self.width])
        self.snake = Snake(self.block_size, self.screen, snake_starting_pos)

        self.running = True
        self.apple = apple(height, width, self.block_size, self.screen, range_of_apple_spawn)

        self.collision = collision(self.apple, self.snake)
        self.apple.spawn_apple()
        self.tick = 0
        self.game_manager = games_manager((self.apple.x, self.apple.y), snake_starting_pos, self.height, self.width,
                                          self.snake, load_tables_from_file)
        self.current_game = self.game_manager.switch_state_table(self.apple.apple_position, self.snake.snake_head,
                                                                 snake_starting_pos)
        self.decide_epsilon_greedy()
        self.games_count = 0
        self.longest_streak = 0
        self.f_approx = f_approximation(self.epsilon)
        self.dqn_agent = DQN_agent(action_number=4, frames=3, learning_rate=0.001, discount_factor=0.95, batch_size=32,
                                   epsilon=1)

        self.debug   = []
    def decide_epsilon_greedy(self):
        if load_tables_from_file[0]:
            self.epsilon = 0
        else:
            self.epsilon = 0.1

    def run(self):
        while self.running:
            self.clockobject.tick(9000)
            self.draw_board()

            self.tick += 1

            reward = self.collision.return_reward(self.height, self.width)

            self.apple.draw_apple()
            self.snake.draw_snake()

            action = self.dqn_agent.make_action(self.get_state(), reward)

            self.process_ai_input(action)
            self.lose_win_scenario(reward)

            pygame.display.flip()

    def lose_win_scenario(self, reward):
        if reward == -1 or reward == 1:
            if reward == 1:
                self.longest_streak += 1
                if self.longest_streak == 110:
                    self.epsilon = 0

            self.games_count += 1
            # print(reward)
            if reward == -1:
                print("Streak to eat apples without dying {}".format(self.longest_streak))
                self.longest_streak = 0
                self.snake.reset_snake()

            self.apple.spawn_apple()
            self.f_approx.restart()

    def draw_board(self):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, (119, 136, 153), rect)

    def get_state(self):
        observation = pygame.surfarray.array3d(pygame.display.get_surface())

        red = pygame.surfarray.pixels_red(pygame.display.get_surface())
        green = pygame.surfarray.pixels_green(pygame.display.get_surface())
        blue = pygame.surfarray.pixels_blue(pygame.display.get_surface())

        inside = np.array([red, green, blue])


        return self.ProcessGameImage(inside)

    def process_ai_input(self, action):
        self.snake.action(action)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Be interpreter friendly
                self.game_manager.save_model("9x9model")
                pygame.quit()
            pygame.display.update()

    def ProcessGameImage(self, RawImage):
        GreyImage = skimage.color.rgb2gray(RawImage)
        # Get rid of bottom Score line
        # Now the Pygame seems to have turned the Image sideways so remove X direction
        CroppedImage = GreyImage[0:400, 0:400]
        # plt.imshow(CroppedImage)
        # print("Cropped Image Shape: ",CroppedImage.shape)
        ReducedImage = skimage.transform.resize(RawImage, (1,3,84, 84), mode='reflect')
        ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range=(0, 255))
        # plt.imshow(ReducedImage)
        # plt.show()
        # Decide to Normalise

        return ReducedImage / 255


snake = Board(6, 6)
snake.run()
