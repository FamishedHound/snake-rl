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

from GAN.model import UNet


class DQN_agent():
    def __init__(self, action_number, frames, learning_rate, discount_factor, batch_size, epsilon, save_model,
                 load_model, path, epsilon_speed):

        self.save_model = save_model
        self.load_model = load_model
        self.epsilon_speed = epsilon_speed

        self.Q_network = DQN(action_number, frames).cuda()
        if self.load_model:
            self.Q_network.load_state_dict(torch.load(path))

        self.target_network = DQN(action_number, frames).cuda()
        self.optimizer_Q_network = torch.optim.Adam(self.Q_network.parameters(), lr=learning_rate)

        self.discount_factor = discount_factor
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.memory = replay_memory(1000000)
        self.epochs = 20

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

        self.gan = UNet(5, 1)
        self.gan = self.gan.cuda()
        self.gan.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1.pt"))

    def update_Q_network(self):
        if len(self.memory.memory) > self.batch_size + 1:

            with torch.enable_grad():
                batch = self.memory.sample(self.batch_size)
                states, actions, rewards, future_states, terminals, terminals_reward = torch.empty(
                    (self.batch_size, self.frames, 84, 84), requires_grad=True).cuda(), torch.empty((self.batch_size),
                                                                                       requires_grad=True).cuda(), torch.empty(
                    (self.batch_size), requires_grad=True).cuda(), torch.empty((self.batch_size, self.frames, 84, 84),
                                                                  requires_grad=True).cuda(), torch.empty((self.batch_size),
                                                                                                          requires_grad=True).cuda(), torch.empty(
                    (self.batch_size), requires_grad=True).cuda()
                for i in range(len(batch)):
                    states[i], actions[i], rewards[i], future_states[i], terminals[i], terminals_reward[i] = batch[i][
                                                                                                                 0], \
                                                                                                             batch[i][
                                                                                                                 1], \
                                                                                                             batch[i][
                                                                                                                 2], \
                                                                                                             batch[i][
                                                                                                                 3], \
                                                                                                             batch[i][
                                                                                                                 4], \
                                                                                                             batch[i][5]

                future_states = future_states.cuda()

                self.Q_network.train()

                self.optimizer_Q_network.zero_grad()
                response = self.Q_network(states)

                loss_input = response
                loss_target = loss_input.clone()

                new_values = rewards + torch.mul(self.target_network(future_states).max(dim=1).values[
                                                     0] * (1 - terminals) + terminals * terminals_reward,
                                                 self.discount_factor)
                new_values = new_values.cuda()

                idx = torch.cat((torch.arange(self.batch_size).float().cuda(), actions)).cuda()
                idx = idx.reshape(2, self.batch_size).cpu().long().numpy()
                loss_target[idx] = new_values

                loss = mse_loss(input=loss_input, target=loss_target)

                # if self.epsilon <= 0.1:
                #                 #     print(""  "")
                #                 #     #plt.plot(self.plot)
                #                 #     #plt.show()
                #                 #     print()

                self.plot.append(loss.item())
                loss.backward()
                self.optimizer_Q_network.step()

    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())  # nie wiem czy tak można ale warto spróbować

    def make_action(self, state, reward, terminal):
        self.Q_network.eval()

        with torch.no_grad():
            state = torch.from_numpy(state.copy()).unsqueeze(0).unsqueeze(0)
            state = state.float().cuda()
            network_response = self.Q_network(state)
            values, indices = network_response.max(dim=1)

            randy_random = random.uniform(0, 1)

            if randy_random > self.epsilon:


                if self.previous_action != None:
                    forbidden_move = self.forbidden_action()
                    network_response[0][forbidden_move] = -99
                    possible_actions = network_response[0]
                    values, indices = possible_actions.max(dim=0)
                    #print(possible_actions)

                    action = indices.item()
                    #print("previous_action {} action {} forbidden_action {}".format(self.previous_action, action,forbidden_move))

                else:
                    action = indices.item()

            else:
                actions = [0, 1, 2, 3]
                if self.previous_action !=None:
                    forbidden_move = self.forbidden_action()
                    actions.remove(forbidden_move)
                action = random.choice(actions)

            self.debug(action)

            self.update_memory(reward, terminal, state,action)

            self.flag = True
            self.previous_action = action
            self.previous_state = state.clone()
            self.previous_reward = reward

            self.sync_networks()
            if terminal:
                if reward == -1:
                    self.previous_action = None
                    self.previous_state = None
                    self.previous_reward = None
                self.flag = False
            return action

    def forbidden_action(self):
        if self.previous_action == 0:
            forbidden_move = 1
        elif self.previous_action == 1:
            forbidden_move = 0
        elif self.previous_action == 2:
            forbidden_move = 3
        elif self.previous_action == 3:
            forbidden_move = 2
        return forbidden_move

    def debug(self, action):
        self.x += 1
        if self.epsilon > 0.1:  # WAS 0.1 CHANGE ME THIS IS TEST !!
            self.epsilon -= self.epsilon_speed

        if self.x % 1111 == 0:
            if self.save_model:
                print("weights saved :) ")
                torch.save(self.Q_network.state_dict(), "DQN_trained_model/10x10_model_with_tail.pt")
            print(self.epsilon)
            print(action)
    #Update memory without GAN
    # def update_memory(self, reward, terminal, state):
    #     if self.flag:
    #         self.memory.append(
    #             (self.previous_state, self.previous_action, self.previous_reward, state, terminal,
    #              reward))
    #
    #         self.update_Q_network()
    def update_memory(self, reward, terminal, state,action):
        import torch
        # action_vec = np.zeros(4)
        # action_vec[action] = 1
        # action = action_vec
        self.gan.eval()
        for x in range(4):

            action_vec = np.zeros(4)
            action_vec[x] = 1
            action_input = action_vec
            action_output = torch.ones_like(torch.from_numpy(state.cpu().numpy().squeeze())).repeat(4, 1, 1) * torch.from_numpy(action_input) \
                .unsqueeze(1) \
                .unsqueeze(2)
            state_action = torch.cat([torch.from_numpy(state.cpu().numpy().squeeze()).unsqueeze(0).cuda(), action_output.float().cuda()], dim=0)
            future_state = self.gan(state_action.unsqueeze(0).cuda())
            self.memory.append(
                (state, action, reward, future_state, terminal,
                 reward))

        self.update_Q_network()

    def sync_networks(self):
        if self.sync_counter % 10 == 0:
            self.update_target_network()
