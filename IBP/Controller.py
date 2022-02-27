from json import load
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from DQN.DQN_agent import replay_memory

class ControllerAgent():
    def __init__(self, action_number, frames, context_size, learning_rate, discount_factor, batch_size, epsilon, save_model,
                 load_model, path, epsilon_speed, cuda_flag=True):
        self.cuda_flag = cuda_flag
        self.save_model = save_model
        self.load_model = load_model
        self.epsilon_speed = epsilon_speed

        self.context_size = context_size

        self.network = ControllerModel(context_size=context_size, output_size=action_number)
        self.target_network = ControllerModel(context_size=context_size, output_size=action_number)

        if self.cuda_flag:
            self.network = ControllerModel(context_size=context_size, output_size=action_number).cuda()
            self.target_network = ControllerModel(context_size=context_size, output_size=action_number).cuda()
        else:
            self.network = ControllerModel(context_size=context_size, output_size=action_number)
            self.target_network = ControllerModel(context_size=context_size, output_size=action_number)

        #if self.load_model:
        #    self.network.load_state_dict(torch.load(path, map_location=('cpu')))

        self.optimizer_network = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        self.discount_factor = discount_factor
        #self.target_network.load_state_dict(self.network.state_dict())
        self.memory = replay_memory(10000)
        self.frames = frames
        self.previous_action = None
        self.previous_state = 0
        self.batch_size = batch_size
        self.sync_counter = 0
        self.epsilon = epsilon
        self.flag = False
        self.previous_reward = None

    def update_network():
        #TODO
        pass
        
    def make_action(self, state, context, reward, terminal):
        self.network.eval()
        with torch.no_grad():
            state = torch.from_numpy(state.copy()).unsqueeze(0).unsqueeze(0)
            context = torch.from_numpy(context)
            if self.cuda_flag:
                state = state.float().cuda()
            else:
                state = state.float()

            network_output = self.network(state)
            values, indices = network_output.max(dim=1)

            randy_random = random.uniform(0, 1)

            if randy_random > self.epsilon:

                 #ASK LUKASZ ABOUT THIS
                if self.previous_action != None:
                    forbidden_move = self.forbidden_action()
                    network_output[0][forbidden_move] = -99
                    possible_actions = network_output[0]
                    values, indices = possible_actions.max(dim=0)

                    action = indices.item()
                else:
                    action = indices.item()

            else:
                actions = [0, 1, 2, 3]
                if self.previous_action != None:
                    forbidden_move = self.forbidden_action()
                    actions.remove(forbidden_move)
                action = random.choice(actions)

            if self.flag:
                self.update_network(reward, terminal, state, context)
            self.flag = True
            self.previous_action = action
            self.previous_state = state.clone()
            self.previous_reward = reward

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

    

class ControllerModel(nn.Module):
    def __init__(self, context_size, state_size=84, hidden_size=512, output_size=1):
        super().__init__()
        # state_size = 84x84
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4) #out=20
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) #out=9
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1) #out=7

        
        self.linear1 = nn.Linear(7 * 7 * 64 + context_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state, context):
        X = F.relu(self.conv1(state))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = X.view(X.size(0), 7 * 7 * 64)
        print("trouble")
        
        X = F.relu(self.linear1(torch.cat(X, context)))
        print("no trouble, amazingly")
        X = self.linear2(X)
        return X

