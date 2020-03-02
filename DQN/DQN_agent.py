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
        self.memory = replay_memory(1000)

        self.previous_action = None
        self.previous_state = 0
        self.batch_size = batch_size
        self.sync_counter = 0
        self.epsilon = epsilon
        self.flag = False

    def update_Q_network(self):
        if len(self.memory.memory) > self.batch_size + 1:

            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, future_states = torch.empty((32, 3, 84, 84)), torch.empty((32)), torch.empty(
                (32)), torch.empty((32, 3, 84, 84))
            for i in range(len(batch)):
                states[i], actions[i], rewards[i], future_states[i] = batch[i][0],batch[i][1],batch[i][2],batch[i][3]




            target = rewards + torch.mul(self.target_network(future_states).max(dim=1).values[
                                            0], self.discount_factor)

            target = Variable(target, requires_grad=True).cuda()

            loss_input = torch.mul(self.Q_network(states)[0][actions.long()], 1)

            loss_input = Variable(loss_input, requires_grad=True).cuda()

            loss = mse_loss(loss_input, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def make_action(self, state, reward):
        state = torch.from_numpy(state.transpose((2, 0, 1))).unsqueeze(0)
        network_response = self.Q_network(state)
        values, indices = network_response.max(dim=1)

        randy_random = random.uniform(0, 1)
        if randy_random > self.epsilon:
            action = indices.item()
        else:
            action = random.choice([0, 1, 2, 3])

        if self.epsilon > 0.1:
            self.epsilon -= 1e-5
        print(" I made action {} with epsilon {} ".format(action, self.epsilon))
        self.update_memory(reward, state)
        self.flag = True
        self.previous_action = action
        self.previous_state = state
        self.sync_networks()
        return action

    def update_memory(self, reward, state):
        if self.flag:
            self.memory.append((self.previous_state, self.previous_action, reward, state))
            self.update_Q_network()
            self.sync_counter += 1

    def sync_networks(self):
        if self.sync_counter % 100 == 0:
            self.update_target_network()
