import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, action_no, how_many_frames):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(how_many_frames, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.tick = 0
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_no)

        # def conv2d_size_out(size, kernel_size=5, stride=2):
        #     return (size - (kernel_size - 1) - 1) // stride + 1
        #
        # # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        # # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        # # linear_input_size = convw * convh * 64
        # self.head = nn.Linear(linear_input_size, action_no)

    def forward(self, observation):
        # if self.training:
        #     print(observation.shape)
        #     print(np.array_equal(observation[0],observation[1]))
        #     print(observation[0])
        #     print(observation[1])
        #     exit(1)

        x = F.relu(self.conv1(observation))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.fc1(x.view(x.size(0), 7 * 7 * 64)))

        x = self.fc2(x)

        return x
