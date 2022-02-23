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
    def __init__(self, dqn_agent, proj_path, state_size, cuda_flag=True):
        self.manager = ManagerModel()
        gan_path = proj_path
        gan_path += f"new_models\\GAN13_3_15_new.pt"
        self.controller = dqn_agent             #LOAD IN BOTH MODELS
        if not cuda_flag:
            self.GAN = torch.load(gan_path, map_location=torch.device('cpu'))
        else:
            self.GAN = torch.load(gan_path)
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
            route = 0 #self.manager.get_route()
            real_reward = env.collision.return_reward(env.height, env.width)
            real_state = env.get_state()

            action = self.select_action(state=real_state, reward=real_reward)
            # Remember, run_step automatically updates internal state of 
            # environment, local state need not be updated with new_state
            # Same goes for reward - both on previous lines are updated
            # by environment methods
            new_state, real_reward, done = env.run_step(action,
                                                        apple_crawl=False)

            # Push "plan context" through LSTM
            # LSTM takes:

            # (After Imagining)
            # manager output (route) p_j_k
            # current (real_state) s_j
            # current (imagined_state, given route - can be s_j again) s_j_pjk
            # action decided (action) a_j,k
            # state imagined (next_imagined_state) s_j_k+1
            # resultant reward (reward) r_j_k
            # j
            # k
            # c_i-1
            #
            # OR
            #
            # (After Acting)
            # manager output (route) p_j_k
            # current (real_state) s_j
            # current (imagined_state "base", just s_j again) s_j_0
            # action decided (action) a_j
            # resultant world state (next_state) s_j+1
            # resultant reward (reward) r_j
            # j
            # k
            # c_i-1 .

            context = self.memory.forward(route=route, 
                                          real_state=real_state, 
                                          imagined_state=real_state,
                                          action=action,
                                          next_state=new_state,
                                          reward=real_reward,
                                          j=j,
                                          k=k,
                                          context=context)

            if real_reward == 10:
                score += 1            

            if done:
                return score

        
        '''
        NOTING DOWN IBP ALGORITHM FOR EASY REFERENCE:

        funciton a^M (x, x*)
            h <- ()
            n_real <- 0
            n_imagined <- 0

            x_real <- x
            x_imagined <- x

            while n_real < n_max_real_steps
                r <- policy_m (x_real, x*, h_n)

                if r == 0 OR n_imagined < n_max_imagined_steps
                    c <- policy_c (x_real, x*, h_n)
                    x_real <- World(x_real, c)
                    n_real += 1
                    n_imagined = 0
                    x_imagined <- x_real

                else if r == 1 
                    c <- policy_c (x_real, x*, h_n)
                    x_imagined <- I(x_real, x*, h_n)
                    n_imagined += 1

                else if r == 2
                    c <- policy_c (x_imagined, x*, h_n)
                    x_imagined <- I(x_imagined, c)
                    n_imagined += 1

                h <- u(h, c, r, x_real, x_imagined, n_real, n_imagined) 
        '''

        pass
