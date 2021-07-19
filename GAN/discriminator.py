# import torch.nn as nn
#
# class DiscriminatorSmall(nn.Module):
#     def __init__(self, ndf):
#         super(DiscriminatorSmall, self).__init__()
#         self.conv1 = nn.Conv2d(1, ndf, 4, 2, padding=1, bias=False)
#         self.relu1 = nn.LeakyReLU(0.2, inplace=True)
#         self.conv2 = nn.Conv2d(ndf, 2*ndf, 4, 2, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(2*ndf)
#         self.relu2 = nn.LeakyReLU(0.2, inplace=True)
#         self.conv3 = nn.Conv2d(2*ndf, 4*ndf, 4, 2, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(4*ndf)
#         self.relu3 = nn.LeakyReLU(0.2, inplace=True)
#         self.conv4 = nn.Conv2d(4*ndf, 1, 3, 1, 0, bias=False)
#         self.avg_pooling = nn.AvgPool2d(8)
#         self.sigmoid4 = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.avg_pooling(x)
#         x = self.sigmoid4(x)
#         return x