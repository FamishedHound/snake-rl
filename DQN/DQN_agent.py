import pickle
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
import matplotlib.pyplot as plt
from GAN.model import UNet
from GAN.reward_model import reward_model
from tree.Node import Node, Master


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
        self.memory = replay_memory(50000)
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
        self.reward_predictor = reward_model(6)
        self.gan = self.gan.cuda()
        self.reward_predictor = self.reward_predictor.cuda()
        self.gan.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1_2frame_with_discriminator.pt"))
        self.reward_predictor.load_state_dict(
            torch.load(
                "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\reward_predictor_future_2frame_new.pt"))
        self.gan.eval()
        self.reward_predictor.eval()
        self.temp_memory = []
        self.observation_counter = 0
        self.loss_plot = []
        self.running = False
        self.action_to_victory = []

    def update_Q_network(self):
        if len(self.memory.memory) > self.batch_size + 1:

            with torch.enable_grad():
                batch = self.memory.sample(self.batch_size)
                states, actions, rewards, future_states, terminals, terminals_reward = torch.empty(
                    (self.batch_size, self.frames, 84, 84), requires_grad=True).cuda(), torch.empty((self.batch_size),
                                                                                                    requires_grad=True).cuda(), torch.empty(
                    (self.batch_size), requires_grad=True).cuda(), torch.empty((self.batch_size, self.frames, 84, 84),
                                                                               requires_grad=True).cuda(), torch.empty(
                    (self.batch_size),
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
                self.loss_plot.append(loss.item())

                self.plot.append(loss.item())
                loss.backward()
                self.optimizer_Q_network.step()
                self.sync_counter += 1

    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def make_action(self, state, reward):
        self.Q_network.eval()
        # plt.imshow(state.squeeze())
        # plt.show()
        terminal = False
        #1 framee change
        if reward == -1 or terminal==1:
            terminal = True
        with torch.no_grad():

            state = torch.from_numpy(state.copy()).unsqueeze(0).unsqueeze(0)
            state = state.float().cuda()
            randy_random = random.uniform(0, 1)
            if len(self.temp_memory) > 0:

                state_memory, _, _ = self.temp_memory[-1]
                state_past = torch.cat([state_memory.squeeze().unsqueeze(0), state.squeeze().unsqueeze(0)])

            else:
                state_past = torch.cat([state.squeeze().unsqueeze(0), state.clone().squeeze().unsqueeze(0)])

            if randy_random > self.epsilon:
                #1 frame change before I used state_past
                dqn_action = self.decide_DQN_action(state)
                action = dqn_action
                #1 frame change


                #action  = self.tree_search(state)

            else:
                actions = [0, 1, 2, 3]
                # removed so the reward predictor can learn failing with a tail
                if self.previous_action != None:
                    forbidden_move = self.forbidden_action(self.previous_action)
                    if forbidden_move != None:
                        actions.remove(forbidden_move)
                action = random.choice(actions)
            if self.previous_action != None:
                self.update_memory_one_frame(state, reward, terminal)
            self.substitute_epsilon(action)

            #self.update_memory(reward, action, terminal, state)

            self.flag = True
            self.previous_action = action
            self.previous_state = state.clone()
            self.previous_reward = reward

            self.sync_networks()
            if terminal:
                if reward == -1:
                    self.clear_temporary_variables()
                self.flag = False
            return action

    def clear_temporary_variables(self):
        self.previous_action = None
        self.previous_state = None
        self.previous_reward = None
        self.temp_memory = []
        self.action_to_victory = []
        self.running = False

    def decide_DQN_action(self, state_past):
        #1 frame change
        network_response = self.Q_network(state_past)
        # forbidden_move = self.forbidden_action()
        # network_response[0][forbidden_move] = -99
        possible_actions = network_response[0]
        values, indices = possible_actions.max(dim=0)
        action = indices.item()
        return action

    def forbidden_action(self, past_action):
        if past_action == 0:
            forbidden_move = 1
        elif past_action == 1:
            forbidden_move = 0
        elif past_action == 2:
            forbidden_move = 3
        elif past_action == 3:
            forbidden_move = 2
        else:
            return None
        return forbidden_move

    def substitute_epsilon(self, action):
        self.x += 1
        if self.epsilon > 0.1:  # WAS 0.1 CHANGE ME THIS IS TEST !!
            self.epsilon -= self.epsilon_speed

        if self.x % 211 == 0:
            # self.show_some_memory()
            if self.save_model:
                print("weights saved :) ")

                torch.save(self.Q_network.state_dict(), "DQN_trained_model/10x10_model_with_tail_new.pt")
            print(self.epsilon)

        self.update_Q_network()
    def update_memory_one_frame(self,state,reward,terminal):
        self.memory.append(
            (self.previous_action, self.previous_action, self.previous_reward, state, terminal,
             reward))
        self.update_Q_network()
    def update_memory(self, reward, action, terminal, current_frame):

        self.temp_memory.append((current_frame, action, reward))
        if len(self.temp_memory) == 3:

            first_frame, first_action, first_reward = self.temp_memory[0]
            second_frame, second_action, second_reward = self.temp_memory[1]

            current_state = torch.cat(
                [first_frame.squeeze().unsqueeze(0), second_frame.squeeze().unsqueeze(0)])
            future_state = torch.cat([second_frame.squeeze().unsqueeze(0), current_frame.squeeze().unsqueeze(0)])
            # plt.imshow(first_frame.squeeze().cpu())
            # plt.show()
            # plt.imshow(second_frame.squeeze().cpu())
            # plt.show()
            # plt.imshow(state.squeeze().cpu())
            # plt.show()

            # print(f" f_r{first_reward} f_a{first_action} s_r{second_reward} s_a{second_action} r{reward} a{action}")

            self.memory.append(
                (current_state, second_action, second_reward, future_state, terminal,
                 reward))

            self.temp_memory.pop(0)
            self.observation_counter += 1
        elif len(self.temp_memory) == 2:

            first_frame, first_action, first_reward = self.temp_memory[-2]

            current_state = torch.cat(
                [first_frame.squeeze().unsqueeze(0), first_frame.squeeze().unsqueeze(0)])
            future_state = torch.cat([first_frame.squeeze().unsqueeze(0), current_frame.squeeze().unsqueeze(0)])

            self.memory.append(
                (current_state, first_action, first_reward, future_state, terminal,
                 reward))
            # plt.imshow(first_frame.squeeze().cpu())
            # plt.show()
            # plt.imshow(state.squeeze().cpu())
            # plt.show()

            self.observation_counter += 1
        self.update_Q_network()

    def tree_search(self, frame):
        self.gan.eval()
        self.reward_predictor.eval()
        master = Master(frame.squeeze(0), self.previous_action)

        root = [master]
        all_nodes = [master]
        self.running = True
        winner_node = None
        while self.running:
            temp = []
            for states in root:
                found = False
                for action_number in range(4):
                    # if torch.equal(master.img.cpu(), states.img.cpu()):
                    #     if self.forbidden_action(master.action) == action_number:
                    #         continue
                    if self.forbidden_action(states.action) == action_number:
                        continue
                    present, future, is_done = self.generate_future(action=action_number, state=states.img)
                    result = Node(parent=states.img, action=action_number, img=future)

                    if present != None:

                        all_nodes.append(result)
                        temp.append(result)

                    if is_done:
                        winner_node = result
                        found = True
                        break
                if found:
                    self.running = False
                    break

            root = temp
        #print(self.back_propagate_tree(all_nodes, winner_node, master.img))
        return self.back_propagate_tree(all_nodes, winner_node, master.img)

    def back_propagate_tree(self, all_nodes: list, winner_node, master):
        path = []
        current_parent = winner_node.parent
        while current_parent != None:
            try:
                action, parent = self.look_for_parent(all_nodes, current_parent, master, winner_node)
                path.append(action)
                current_parent = parent
            except Exception as e:
                break
        if len(path)==1:
            return path[0]
        else:
            return path[::-1][1]


    def look_for_parent(self, all_nodes, suspect, master_img, winner_node):
        for node in all_nodes:

            if torch.equal(master_img.cpu(), suspect.cpu()):
                return winner_node.action, None
            if torch.equal(node.img.cpu(), suspect.cpu()):
                return node.action, node.parent

    def generate_future(self, action, state):
        self.gan.eval()
        self.reward_predictor.eval()
        with torch.no_grad():
            action_vec = np.zeros(4)
            action_vec[action] = 1
            action = action_vec

            action = torch.ones_like(state.squeeze().cuda()).repeat(4, 1, 1) * torch.from_numpy(action).cuda() \
                .unsqueeze(1) \
                .unsqueeze(2)

            state_action = torch.cat([state, action.float()])
            future_state = self.gan(state_action.unsqueeze(0))
            future_state = future_state.squeeze(0)
            # plt.imshow(state.cpu().squeeze(),cmap='gray',vmax=1,vmin=0)
            # plt.show()
            #
            #
            # plt.imshow(future_state.cpu().squeeze(),cmap='gray',vmax=1,vmin=0)
            # plt.show()

            two_frame = torch.cat([state, future_state])
            reward = self.determine_reward(self.reward_predictor(two_frame.unsqueeze(0)))
            # print(reward)
            if reward == -1:
                return None, future_state.cuda(), True if reward == 1 else False

            return state, future_state.cuda(), True if reward == 1  else False
            # plt.imshow(future_state.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
            # plt.show()

        # self.memory.append(
        #     (self.previous_state, self.previous_action, self.previous_reward, state, terminal,
        #      reward))

    def determine_reward(self, reward):
        # mapping of reward 0 => 10 , 1 => -1 ,  2 => -0.1

        rewards = [1, -1, 0]
        reward_in = torch.argmax(reward).item()

        return rewards[reward_in]

    def show_some_memory(self):
        for x in self.memory.sample(10):
            (state, which_action, reward, future_state, terminal, terminal_reward) = x
            plt.imshow(state[0].cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
            plt.show()
            plt.imshow(state[1].cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
            plt.show()
            plt.imshow(future_state[0].cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
            plt.show()
            plt.imshow(future_state[1].cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
            plt.show()
            print(f"a{which_action}, r{reward}, is_t{terminal_reward} , t_r{terminal_reward}")

    def sync_networks(self):
        if self.sync_counter % 5 == 0:
            self.update_target_network()
