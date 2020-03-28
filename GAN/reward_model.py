import torch.nn.functional as F

from GAN.gan_utils import *

class reward_model(nn.Module):
    def __init__(self, n_channels):
        super(reward_model, self).__init__()
        # Reward predictor
        self.conv1 = nn.Conv2d(n_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=1)
        self.fn1 = nn.Linear(5 * 16, 512)
        self.fn2 = nn.Linear(512, 512)
        self.fn3 = nn.Linear(512, 3)
        self.max_pl = nn.MaxPool2d(kernel_size=21, stride=21)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        reward = self.max_pl(x)
        reward = reward.reshape(reward.size(0), -1)
        reward = F.relu(self.fn1(reward))
        reward = F.relu(self.fn2(reward))
        reward = self.fn3(reward)

        return F.softmax(reward,dim=1)