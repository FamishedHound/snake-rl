import pickle
from scipy import ndimage, misc
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
import cv2
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

snake_starting_pos = (1, 3)
# ( bool (do we want to load) ,  filename )
load_tables_from_file = (False, "9x9model")

range_of_apple_spawn = (1, 2)


class Board():
    def __init__(self, height, width):
        self.clockobject = pygame.time.Clock()

        pygame.init()
        self.block_size = 50
        self.height = height
        self.width = width
        self.screen = pygame.display
        self.screen.set_mode([self.block_size * self.height, self.block_size * self.width])
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
        self.dqn_agent = DQN_agent(action_number=4, frames=1, learning_rate=0.001, discount_factor=0.99, batch_size=32,
                                   epsilon=1, save_model=True, load_model=False, path="",epsilon_speed=1e-4)
        self.reward = 0
        self.action = None
        self.speed = 9000
        self.debug = []

    def decide_epsilon_greedy(self):
        if load_tables_from_file[0]:
            self.epsilon = 0
        else:
            self.epsilon = 0.1

    def run(self):

        while self.running:
            self.games_count += 1
            pygame.display.flip()
            reward = self.collision.return_reward(self.height, self.width)
            self.clockobject.tick(self.speed)
            self.draw_sprites()
            self.process_input()

            action = self.dqn_agent.make_action(self.get_state(), reward,
                                                True if reward == 1 or reward == -1 else False)

            self.snake.action(action)
            self.tick += 1
            self.lose_win_scenario(reward)

    def draw_sprites(self):
        self.draw_board()
        self.apple.draw_apple()
        self.snake.draw_snake()

    def lose_win_scenario(self, reward):
        if reward == -1 or reward == 1:

            if reward == -1:
                print("Apple score {}".format(self.longest_streak))
                self.longest_streak = 0
                self.snake.reset_snake()
            if reward == 1:
                self.longest_streak += 1
                if self.longest_streak == 110:
                    self.epsilon = 0

            # print(reward)

            # Change me if you want random apple

    def draw_board(self):
        for y in range(self.height):
            for x in range(self.width):
                # Rectangle drawing
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen.get_surface(), (0, 0, 0), rect)

    def get_state(self):

        # I take red blue green RGB
        red = pygame.surfarray.pixels_red(self.screen.get_surface())
        green = pygame.surfarray.pixels_green(self.screen.get_surface())
        blue = pygame.surfarray.pixels_blue(self.screen.get_surface())

        inside = np.array([red, green, blue]).transpose((1, 2, 0))

        return self.ProcessGameImage(inside)

    def process_input(self):

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
        # RawImage = GreyImage[0:400, 0:400]
        # plt.imshow(GreyImage)
        # plt.show()
        # print("Cropped Image Shape: ",CroppedImage.shape)

        # ReducedImage = skimage.transform.resize(RawImage, (84, 84,3), mode='reflect')
        # ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range=(0, 255))
        img = cv2.resize(GreyImage, (84, 84))

        # plt.imshow(img.astype(np.uint8))
        # plt.show()
        # print(img.shape)
        img = ndimage.rotate(img, 270, reshape=False)
        # plt.imshow(img)
        # plt.savefig(f"./output/{self.games_count}.png")
        # plt.close()
        return img


snake = Board(4, 4)
snake.run()
