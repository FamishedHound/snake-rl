import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
import os
from torchvision.transforms import functional as TF
from torchvision import transforms
import torch


class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None, num_images=1):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index]
        img = Image.open(img)


        if self.transform is not None:
            img = self.transform(img)

        actions = torch.ones_like(img1).repeat(1,4,1,1) * torch.from_numpy(actions).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        state_action = torch.cat([state, action], dim=1)
        return


class DeblurDataset(object):
    def __init__(self, data_path):
        data = []
        for folder in os.listdir(osp.join(data_path, 'train')):
            for fname in os.listdir(osp.join(osp.join(data_path, 'train_sharp/train/train_sharp', folder))):
                data.append(osp.join(data_path, 'train', folder, fname))
        self.data_train = data
        data = []
        for folder in os.listdir(osp.join(data_path, 'val')):
            for fname in os.listdir(osp.join(osp.join(data_path, 'val_sharp/val/val_sharp', folder))):
                data.append(osp.join(data_path, 'val', folder, fname))
        self.data_val = data


if __name__ == '__main__':
    data_path = '/home/galidor/Documents/PredictiveFilterFlow/datasets'
    dset = DeblurDataset(data_path)
    print(len(dset.data))
from torchvision import transforms
train_transforms_list = [transforms.ToTensor(),
                             # transforms.Normalize(mean, std)
                             ]
from torch.utils.data import DataLoader
# train_transforms_list = [transforms.ToTensor(),
#                          transforms.ToPILImage()]
train_transforms = transforms.Compose(train_transforms_list)
data_train = ImageDataset(DeblurDataset(data_path).data_train, transform=train_transforms)

data_train_loader = DataLoader(data_train, batch_size=56, shuffle=True, num_workers=16)

for epoch in range(200):
    model.train()
    running_loss = 0.0
    for i, img in enumerate(data_train_loader):
        img_sharp, img_blur = img
        optimizer.zero_grad()
        img_sharp = img_sharp.cuda()
        img_blur = img_blur.cuda()
        img_deblur = net(img_blur)
        loss = XXX(img_deblur, img_sharp)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    scheduler.step()

actions =
