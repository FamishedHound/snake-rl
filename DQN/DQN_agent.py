import random

from DQN.model import DQN

import torch.optim as optim
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.functional import mse_loss

from DQN.replay_memory import replay_memory


class DQN_agent():
    def __init__(self, action_number, frames, learning_rate, discount_factor, batch_size, epsilon):

        self.Q_network = DQN(action_number, frames).cuda()
        self.target_network = DQN(action_number, frames).cuda()
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.memory = replay_memory(1000000)

        self.previous_state = None
        self.batch_size = batch_size
        self.sync_counter = 0
        self.epsilon = epsilon

    def update_Q_network(self):
        if self.memory.memory.__sizeof__() > self.batch_size:
            batch = self.memory.sample(self.batch_size)

            for memory in batch:
                for state, action, reward, future_state in memory:
                    state = torch.from_numpy(state) / 255
                    action = action
                    reward = reward
                    future_state = torch.from_numpy(future_state) / 255

                    target = reward + self.discount_factor * self.target_network(future_state).max(dim=1).values[
                        0].item()
                    loss_input = self.Q_network(state)[0][action].item()

                    loss = mse_loss(loss_input, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def make_action(self, state, reward):
        values, indices = self.Q_network(state).max(dim=1)

        randy_random = random.uniform(0, 1)
        if randy_random > self.epsilon:
            action = indices.item()
        else:
            action = random.choice([range(4)])

        self.update_memory(reward, state, action)
        self.previous_state = state
        self.sync_networks()
        return action

    def update_memory(self, reward, state, action):
        if self.previous_state != None:
            self.memory.append((self.previous_state, action, reward, state))
            self.update_Q_network()
            self.sync_counter += 1

    def sync_networks(self):
        if self.sync_counter % 50:
            self.update_target_network()

        if self.epsilon > 0.1:
            self.epsilon -= 0.1
