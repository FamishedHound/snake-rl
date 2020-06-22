import torch.nn.functional as F

from GAN.gan_utils import *


class reward_model(nn.Module):
    def __init__(self, n_channels):
        super(reward_model, self).__init__()
        # Reward predictor
        # self.conv1 = nn.Conv2d(n_channels, 32, 8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, 3, stride=2)
        # self.conv4 = nn.Conv2d(64, 64, 4, stride=1)
        self.max_pl = nn.MaxPool2d(kernel_size=7, stride=7)
        self.conv1 = nn.Conv2d(2, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 0)
        self.conv3 = nn.Conv2d(128, 512, 3, 2, 0)
        self.max_pl2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fn1 = nn.Linear(512, 512)
        self.fn2 = nn.Linear(512, 512)
        self.fn3 = nn.Linear(512, 3)



        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(512)


    def forward(self, x):
        reward = self.max_pl(x)
        reward = F.relu(self.conv1(reward))
        reward = F.relu(self.conv2(reward))
        reward = F.relu(self.conv3(reward))
        reward = self.max_pl2(reward)
        reward = reward.reshape(reward.size(0), -1)
        reward = F.relu(self.fn1(reward))
        reward = F.relu(self.fn2(reward))
        reward = self.fn3(reward)

        return reward
