import torch.nn.functional as F

from GAN.gan_utils import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Reward predictor
        self.conv1 = nn.Conv2d(n_classes, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(1)
        # Unet
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        reward = F.relu(self.bn1(self.conv1(x)))

        reward = F.relu(self.bn1(self.conv2(reward)))

        reward = F.relu(self.bn1(self.conv3(reward)))

        reward = F.relu(self.bn1(self.conv4(reward)))

        reward = F.relu(self.bn1(self.conv5(reward)))

        return logits, reward
