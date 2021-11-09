import gym
import time
import torch.nn as nn
import matplotlib as plt
from IBP import IBP

env = gym.make('LunarLander-v2')
obs_space = env.observation_space.shape[0] #8
action_space = env.action_space.n #4

img_layers = nn.Sequential(
    nn.Linear(action_space, 64),
    nn.Linear(64, obs_space)
)

ibp = IBP(env, img_layers)

num_eps = 5
flag = True

for i_ep in range(num_eps):
    state_real = env.reset()
    state_imagined = state_real

    #Reset memory

    while flag:

        #Select route

        #Select Action

        #If imagination -> aciton onto imagination

        #If not imagination -> action onto world

        #Concaticate onto memory


        action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        env.render()
        time.sleep(1/30)

        if done:
            break

