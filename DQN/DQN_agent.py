import random

from DQN.model import DQN

import torch.optim as optim
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
from DQN.replay_memory import replay_memory
import numpy as np
from torch.autograd import Variable


class DQN_agent():
    def __init__(self, action_number, frames, learning_rate, discount_factor, batch_size, epsilon):

        self.Q_network = DQN(action_number, frames).cuda()
        self.target_network = DQN(action_number, frames).cuda()
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.memory = replay_memory(1000000)

        self.previous_state = 0
        self.batch_size = batch_size
        self.sync_counter = 0
        self.epsilon = epsilon
        self.flag = False

    def update_Q_network(self):
        if len(self.memory.memory) > self.batch_size + 1:
            batch = self.memory.sample(self.batch_size)

            for memory in batch:
                (state, action, reward, future_state) = memory
                state = torch.from_numpy(state)
                action = action
                reward = reward
                future_state = torch.from_numpy(future_state)

                target = reward + torch.mul(self.target_network(future_state).max(dim=1).values[
                                                0], self.discount_factor)

                target = Variable(target, requires_grad=True).cuda()

                loss_input = torch.mul(self.Q_network(state)[0][action], 1)
                loss_input = Variable(loss_input, requires_grad=True).cuda()

                loss = mse_loss(loss_input, target)
                #print("loss {}".format(loss))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def make_action(self, state, reward):
        network_response = self.Q_network(torch.from_numpy(state))
        values, indices = network_response.max(dim=1)

        randy_random = random.uniform(0, 1)
        if randy_random > self.epsilon:
            action = indices.item()
        else:
            action = random.choice([0, 1, 2, 3])

        if self.epsilon > 0.1:
            self.epsilon -= 1e-5
        print(" I made action {} with epsilon {} ".format(action, self.epsilon))
        self.update_memory(reward, state, action)
        self.flag = True
        self.previous_state = state
        self.sync_networks()
        return action

    def update_memory(self, reward, state, action):
        if self.flag:
            self.memory.append((self.previous_state, action, reward, state))
            self.update_Q_network()
            self.sync_counter += 1

    def sync_networks(self):
        if self.sync_counter % 1000==0:
            self.update_target_network()
