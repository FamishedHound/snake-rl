import sys
from matplotlib import pyplot as plt
sys.path.append("..")
from DQN.DQN_agent import DQN_agent
import torch.nn as nn
import torch
from IBP.Manager import ManagerModel
from IBP.Controller import ControllerModel
from IBP.Memory import LSTMModel
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)


# torch.save(model.state_dict(), 
# proj_path + 
# f"new_models\\GAN13_{generator_amplifier}_{discriminator_deamplifier}_new.pt")

class IBP(object):
    def __init__(self, dqn_agent):
        self.manager = ManagerModel()
        gan_path = "C:\\Users\\killi\\Documents\\Repositories\\snake-rl\\"
        gan_path += f"new_models\\GAN13_3_15_new.pt"
        self.controller = dqn_agent                         #LOAD IN
        self.GAN = torch.load(gan_path)                     #BOTH MODELS
        # self.reward_predictor = ### Our best rew.prededitor
        # this is likely to just be our controller but excluding everything but 
        # the reward - may not need it?
        self.memory = LSTMModel()


    def select_action(self, state, reward):
        return self.controller.make_action(state, reward,
                        True if reward == -1 or reward == 10 else False)

    def plot_results(self, scores):
        plt.figure(figsize=(12,5))
        plt.title("Rewards")
        plt.plot(scores, alpha=0.6, color='red')
        plt.savefig("Snake_Rewards_plot.png")
        plt.close()

    def run(self, env):
        history = []
        num_real = 0
        num_imagined = 0
        score = 0
        while True:
            
            reward = env.collision.return_reward(env.height, env.width)
            state = env.get_state()
            action = self.select_action(state=state, reward=reward)
            # Remember, run_step automatically updates internal state of 
            # environment, local state need not be updated with new_state
            # Same goes for reward - both on previous lines are updated
            # by environment methods
            new_state, reward, done = env.run_step(action, apple_crawl=False)
            if reward == 10:
                score += 1            

            if done:
                return score

        pass
