import pickle
import numpy as np
from PIL.ImageFile import ImageFile
from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torch.optim.adam import Adam
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
import os
from torchvision.transforms import functional as TF
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from GAN.model import UNet
from torch.utils.data import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None, num_images=1, val=False):
        self.dataset = dataset
        self.transform = transform
        self.val = val

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not self.val:
            path = self.dataset[
                index]  # Remember that S_images has 1 image more than Sa_images because index ffor Sa is index-1
            path_output = self.dataset[index].replace("S_images", "Sa_images")

            (pickled_arr,reward) = pickle.load(open(path, "rb"))
            pickled_arr_output = pickle.load(open(path_output, "rb"))

            # if self.transform is not None:
            #     img = self.transform(img)

            return pickled_arr, pickled_arr_output,reward


class DeblurDataset(object):
    def __init__(self, data_path):
        self.data_val = []
        self.data_train = []
        self.data_path = data_path

    def get_paths(self):

        for folder in os.listdir(osp.join(self.data_path)):
            for fname in os.listdir(self.data_path + "\\" + folder):
                if folder == "Sa_images":

                    self.data_val.append(osp.join(self.data_path, folder, fname))
                else:
                    self.data_train.append(osp.join(self.data_path, folder, fname))

        return self


def train_gan():
    from torchvision import transforms
    train_transforms_list = [transforms.ToTensor(),
                             # transforms.Normalize(mean, std)
                             ]
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    train_transforms_list = [transforms.ToTensor(),
                             transforms.ToPILImage()]
    train_transforms = transforms.Compose(train_transforms_list)
    data_train = ImageDataset(DeblurDataset(data_path).get_paths().data_train, transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=16)
    model = UNet(5, 1).cuda()
    #model.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1.pt"))
    optimizer = SGD(model.parameters(), lr=0.01,momentum=0.9)
    optimizer_reward = Adam(model.parameters(), lr=0.0005)
    plot = []
    plot_reward = []
    last_deblurs = None
    for epoch in range(30):
        model.train()
        running_loss = 0.0
        running_loss_reward = 0.0
        for i, img in enumerate(data_train_loader):
            img_blur, img_sharp, reward_actual  = img
            optimizer.zero_grad()
            reward_actual = reward_actual.float().cuda()

            img_sharp = img_sharp.float().cuda()
            img_blur = img_blur.float().cuda()
            img_deblur, reward = model(img_blur)


            loss_reward = BCEWithLogitsLoss()(reward , reward_actual)
            loss = mse_loss(img_deblur, img_sharp)
            loss_combined = loss + loss_reward


            running_loss += loss.item()
            running_loss_reward += loss_reward.item()
            plot.append(loss.item())
            plot_reward.append(loss_combined.item())
            loss_combined.backward()

            print(f"loss image {running_loss / (i + 1)}")
            print(f"loss_reward {running_loss_reward/ (i + 1)}")
            optimizer.step()
            # if i == len(data_train_loader)-1:
            #     last_deblurs = img_deblur
        # plt.plot(plot)
        # plt.show()
        if epoch % 5==0:
            torch.save(model.state_dict(), "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1.pt")
    plt.plot(plot_reward)
    plt.show()
    print(f"finished epoch {epoch}")
    input_image = img_blur[0][0].detach().cpu().numpy().squeeze()
    predicted_output_img = img_deblur[0].detach().cpu().numpy().squeeze()
    actual_output = img_sharp[0].detach().cpu().numpy().squeeze()
    plt.imshow(input_image)
    print("actual reward {} predicted reward {}".format(reward_actual,reward))

    plt.show()
    plt.imshow(predicted_output_img)
    plt.show()
    plt.imshow(actual_output)
    plt.show()
    torch.save(model.state_dict(), "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1.pt")
    plt.plot(plot)
    plt.show()


def validate_prepare_data():
    pickled_arr = pickle.load(
        open("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\validate_gan\\state_s_188.pickle", "rb"))
    # pickled_arr_output = pickle.load(
    #    open("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\validate_gan\\state_s_1.pickle", "rb"))

    return (pickled_arr, None)


def validate_gan():
    train_transforms_list = [transforms.ToTensor(),
                             transforms.ToPILImage()]
    train_transforms = transforms.Compose(train_transforms_list)
    data_train = ImageDataset(DeblurDataset(data_path).get_paths().data_train, transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=False, num_workers=16)

    model = UNet(5, 1).cuda()
#    model.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1.pt"))
    model.eval()
    # for i, img in enumerate(data_train_loader):
    img_blur, img_sharp = validate_prepare_data()
    img_deblur, reward = model(img_blur.unsqueeze(0).float().cuda())

    # input_image = img_blur[0][0].detach().cpu().numpy().squeeze()
    predicted_output_img = img_deblur[0].detach().cpu().numpy().squeeze()
    # actual_output = img_sharp[0].detach().cpu().numpy().squeeze()
    # plt.imshow(input_image)
    # plt.show()
    plt.imshow(predicted_output_img)
    plt.show()
    # plt.imshow(actual_output)
    # plt.show()


if __name__ == '__main__':
    bla = 1
    data_path = "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\train\\"
    # train,val = DeblurDataset(data_path).get_paths()

    train_gan()
    #validate_gan()
