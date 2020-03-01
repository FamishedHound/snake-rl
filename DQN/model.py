import torch
import torch.nn as nn

import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_no, how_many_frames):
        super().__init__()
        self.conv1 = nn.Conv2d(how_many_frames, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_no)

    def forward(self, observation):
        x = F.relu(self.conv1(observation.float().cuda()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), 7 * 7 * 64)))
        x = self.fc2(x)

        return x


