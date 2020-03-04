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
        self.optimizer_Q_network = torch.optim.Adam(self.Q_network.parameters(), lr=learning_rate)

        self.discount_factor = discount_factor
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.memory = replay_memory(10000)

        self.frames = frames
        self.previous_action = None
        self.previous_state = 0
        self.batch_size = batch_size
        self.sync_counter = 0
        self.epsilon = epsilon
        self.flag = False
        self.previous_reward = None
        self.x = 0
        self.plot = []

    def update_Q_network(self):
        if len(self.memory.memory) > self.batch_size + 1:
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, future_states, terminals, terminals_reward = torch.empty(
                (32, self.frames, 84, 84)), torch.empty((32)), torch.empty(
                (32)), torch.empty((32, self.frames, 84, 84)), torch.empty((32)), torch.empty((32))
            for i in range(len(batch)):
                states[i], actions[i], rewards[i], future_states[i], terminals[i], terminals_reward[i] = batch[i][0], \
                                                                                                         batch[i][1], \
                                                                                                         batch[i][2], \
                                                                                                         batch[i][3], \
                                                                                                         batch[i][4], \
                                                                                                         batch[i][5]

            # argmaxs = torch.mul(self.target_network(future_states).max(dim=1).values[
            #                                  0] * (1- terminals) + terminals * terminals_reward, self.discount_factor)
            # target = rewards + argmaxs
            #
            # target = Variable(target, requires_grad=True).cuda()
            #
            # temp = self.Q_network(states)
            # loss_input = torch.mul(temp[0][actions.long()], 1)
            self.Q_network.train()

            self.optimizer_Q_network.zero_grad()
            response = self.Q_network(states)
            loss_input = response
            loss_target = loss_input.clone()

            new_values = rewards + torch.mul(self.target_network(future_states).max(dim=1).values[
                                              0] * (1- terminals) + terminals * terminals_reward, self.discount_factor)
            new_values = new_values.cuda()

            idx = torch.cat((torch.arange(32).float(), actions))
            idx = idx.reshape(2, 32).long().numpy()
            loss_target[idx] = new_values

            loss = mse_loss(input=loss_input, target=loss_target)

            if self.epsilon <= 0.1:
                print(""  "")
                plt.plot(self.plot)
                plt.show()
                print()

            self.sync_counter += 1
            self.plot.append(loss.item())
        
            loss.backward()
            self.optimizer_Q_network.step()

            # learning_rate = 0.01
            # with torch.no_grad():
            #
            #     for param in self.Q_network.parameters():
            #         param -= learning_rate * param.grad

    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict().clone()) #nie wiem czy tak można ale warto spróbować

    def make_action(self, state, reward, terminal):
        self.Q_network.eval()
        with torch.no_grad():
            state = torch.from_numpy(state.copy()).unsqueeze(0).unsqueeze(0)

            network_response = self.Q_network(state)
            values, indices = network_response.max(dim=1)

            randy_random = random.uniform(0, 1)

            # if randy_random > self.epsilon:
            #     action = indices.item()
            #     if self.epsilon <= 0.1:
            #         print(action)
            # else:
            #     action = random.choice([0, 1, 2, 3])
            action = 2
            self.debug(action)

            self.update_memory(reward, terminal, state)

            self.flag = True
            self.previous_action = action
            self.previous_state = state.clone()
            self.previous_reward = reward

            self.sync_networks()
            if terminal:
                self.previous_action = None
                self.previous_state = None
                self.previous_reward = None
                self.flag = False
            return action

    def debug(self, action):
        if self.epsilon > 0.1:
            self.epsilon -= 1e-4
            self.x += 1
        if self.x % 1111 == 0:
            print(self.epsilon)
            print(action)

    def update_memory(self, reward, terminal, state):
        if self.flag:
            self.memory.append(
                (self.previous_state.detach(), self.previous_action, self.previous_reward, state.detach(), terminal,
                 reward))

            self.update_Q_network()

    def sync_networks(self):
        if self.sync_counter % 10 == 0:
            self.update_target_network()
