import torch
import numpy as np
import pygame
from DQN.DQN_agent import DQN_agent
from Function_approx.approximator import f_approximation
from gym_snake.gym_snake.envs.apple import apple
from gym_snake.gym_snake.envs.collision_handler import collision
from gym_snake.gym_snake.envs.games_manager import games_manager
from gym_snake.gym_snake.envs.snake import Snake
import skimage as skimage
from skimage import color
import cv2
import gym
snake_starting_pos = (1, 3)
# ( bool (do we want to load) ,  filename )
load_tables_from_file = (False, "9x9model")

range_of_apple_spawn = (1, 2)


class Board(gym.Env):
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
        self.apple = apple(height, width, self.block_size, self.screen, range_of_apple_spawn, self.snake)

        self.index = 1
        self.past = None
        self.index_helper = 0
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
        # self.dqn_agent = DQN_agent(action_number=4, frames=2, learning_rate=0.01, discount_factor=0.99, batch_size=48,
        #                            epsilon=0.1, save_model=False, load_model=True,
        #                            path="/DQN_trained_model/10x10_model_with_tail_new.pt",
        #                            epsilon_speed=1e-7, snake=self.snake)
        self.reward = 0
        self.action = None
        self.speed = 9000
        self.debug = []
        self.previous_gan_action = None

    def decide_epsilon_greedy(self):
        if load_tables_from_file[0]:
            self.epsilon = 0
        else:
            self.epsilon = 0.1

    # def run(self):
    #
    #     while self.running:
    #         pygame.display.flip()
    #         reward = self.collision.return_reward(self.height, self.width)
    #         self.clockobject.tick(self.speed)
    #
    #         self.draw_sprites()
    #
    #         self.process_input()
    #         self.snake.move_segmentation()
    #         self.snake.draw_segment()
    #         img = self.get_state()
    #
    #         # action = self.dqn_agent.make_action(img, reward,
    #         #                                     True if reward == -1  else False)  # was if reward == 10 or
    #         # #old DQN
    #         action = self.dqn_agent.make_action(img, reward,
    #                                             True if reward == -1 or reward == 10 else False)
    #         self.create_actions_channels(action, img, reward)
    #
    #         self.snake.action(action)
    #
    #         self.tick += 1
    #         self.lose_win_scenario(reward)
    #         self.games_count += 1
    def step(self, action):
        pygame.display.flip()
        self.reward = self.collision.return_reward(self.height, self.width)
        self.clockobject.tick(self.speed)

        self.draw_sprites()

        self.process_input()
        self.snake.move_segmentation()

        self.snake.draw_segment()
        img = self.get_state()



        self.create_actions_channels(action, img, self.reward)

        self.snake.action(action)

        self.tick += 1

        self.games_count += 1
        self.create_actions_channels(action, img, self.reward)

        return [img, self.reward, True if self.reward==-1 else False, None]
    def reset(self):
        self.lose_win_scenario()
        img = self.get_state()
        return img,self.reward
    def draw_sprites(self):
        self.draw_board()
        self.apple.draw_apple()
        self.snake.draw_snake()

    def lose_win_scenario(self):
        if self.reward == -1 :

            if self.reward == -1:
                print("Apple score {}".format(self.longest_streak))
                self.longest_streak = 0
                self.snake.reset_snake()
            if self.reward == 10:
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

        inside = np.array([red, green, blue]).transpose((2, 1, 0))
        # plt.imshow(inside)
        # plt.show()
        return self.ProcessGameImage(inside)

    def process_input(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Be interpreter friendly
                self.game_manager.save_model("9x9model")
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                # print("hgere")
                action = None
                if event.key == pygame.K_RIGHT:
                    pass
                if event.key == pygame.K_UP:
                    self.speed = 9000

                if event.key == pygame.K_LEFT:
                    # print("kkk")
                    action = 0
                if event.key == pygame.K_DOWN:
                    self.speed = 1
                # print(action)

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
        # img = ndimage.rotate(img, 270, reshape=False)
        # if self.index > 0:
        #     plt.imshow(img)
        #     plt.savefig(f"S'_images/{self.index-1}.png")
        #     plt.close()
        #
        #
        #
        # plt.imshow(img)
        # plt.savefig(f"C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\train\\S_images\\{self.index}.png")
        # plt.close()
        # self.index += 1

        return img

    def create_actions_channels(self, action, img, reward):

        # plt.imshow(img,cmap='gray',vmax=1,vmin=0)
        # plt.show()

        action_vec = np.zeros(4)
        action_vec[action] = 1
        action = action_vec
        np_reward = np.zeros(3)
        # mapping of reward 0 => 10 , 1 => -1 ,  2 => -0.1
        if reward == 10:
            np_reward[0] = 1

        elif reward == -1:
            np_reward[1] = 1

        else:
            np_reward[2] = 1

        np_reward = torch.from_numpy(np_reward)
        # print("reward {} vector {}".format(reward, np_reward))
        action = torch.ones_like(torch.from_numpy(img)).repeat(4, 1, 1) * torch.from_numpy(action) \
            .unsqueeze(1) \
            .unsqueeze(2)

        # state_action = torch.cat([torch.from_numpy(img).unsqueeze(0), action], dim=0)
        # state = torch.from_numpy(img)
        # if self.index > 0:
        #     with open(f"train_reward/future/state_s_{self.index - 1}.pickle", 'wb') as handle:
        #
        #         pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(f'train_reward/now/state_s_{self.index}.pickle', 'wb') as handle:
        #     pickle.dump((state, np_reward), handle, protocol=pickle.HIGHEST_PROTOCOL)

        # if self.index % 2 == 0:
        #     # if reward == 10:
        #     #     plt.imshow(self.past, cmap='gray', vmax=1, vmin=0)
        #     #     plt.show()
        #     #     plt.imshow(state, cmap='gray', vmax=1, vmin=0)
        #     #     plt.show()
        #     #     print()
        #     with open(f'train_reward/now/state_s_{self.index}.pickle', 'wb') as handle:
        #         pickle.dump((self.past, state, np_reward), handle, protocol=pickle.HIGHEST_PROTOCOL)
        # self.past = state
        # # GAN
        # if self.index > 0:
        #     with open(f"train/Sa_images/state_s_{self.index - 1}.pickle", 'wb') as handle:
        #         future_state = torch.from_numpy(img).unsqueeze(0)
        #         pickle.dump(future_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(f'train/S_images/state_s_{self.index}.pickle', 'wb') as handle:
        #     pickle.dump((state_action,np_reward), handle, protocol=pickle.HIGHEST_PROTOCOL)

        # generate validation images
        # with open(f'validate_gan/state_s_{self.index}.pickle', 'wb') as handle:
        #  pickle.dump((state_action,reward), handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.previous_gan_action = action
        self.index += 1



