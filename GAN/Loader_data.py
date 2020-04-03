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

from GAN.reward_model import reward_model

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

            pickled_arr, _ = pickle.load(open(path, "rb"))
            pickled_arr_output = pickle.load(open(path_output, "rb"))

            # if self.transform is not None:
            #     img = self.transform(img)

            return pickled_arr, pickled_arr_output


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


class RewardDataset(Dataset):

    def __init__(self, dataset, transform=None, num_images=1, val=False):
        self.dataset = dataset
        self.transform = transform
        self.val = val

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not self.val:
            path = self.dataset[
                index]
            path_future = self.dataset[
                index].replace("now", "future")
            (state, reward) = pickle.load(open(path, "rb"))
            # plt.imshow(state, cmap='gray')
            # plt.show()

            future = pickle.load(open(path_future, "rb"))

            # plt.imshow(future, cmap='gray')
            # plt.show()
            state_future = torch.cat([state.unsqueeze(0), future.unsqueeze(0)], 0)
            return state_future, reward


class RewardPathsDataset(object):
    def __init__(self, data_path):
        self.data_val = []
        self.data_train = []
        self.data_path = data_path

    def get_paths(self):

        for folder in os.listdir(osp.join(self.data_path)):
            for fname in os.listdir(self.data_path + "\\" + folder):
                self.data_train.append(osp.join(self.data_path, folder, fname))

        return self


def train_reward_model():
    from torchvision import transforms
    train_transforms_list = [transforms.ToTensor(),
                             # transforms.Normalize(mean, std)
                             ]
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    train_transforms_list = [transforms.ToTensor(),
                             transforms.ToPILImage()]
    train_transforms = transforms.Compose(train_transforms_list)
    data_train = RewardDataset(balance_files(), transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=16)

    model = reward_model(5).cuda()
    model.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\reward_predictor.pt"))
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    plot = []
    for epoch in range(20):
        model.train()
        running_loss = 0.0

        for i, img in enumerate(data_train_loader):
            state, actual_reward = img
            optimizer.zero_grad()

            actual_reward = actual_reward.float().cuda()
            state = state.float().cuda()
            models_reward = model(state)
            actual_reward = torch.argmax(actual_reward, dim=1)
            loss_reward = CrossEntropyLoss()(models_reward, actual_reward.long())

            running_loss += loss_reward.item()

            plot.append(loss_reward.item())

            loss_reward.backward()
            print("predicted reward {} actual reward {}".format(torch.argmax(models_reward[0]).item(),actual_reward[0].item()))
            print(f"loss image {running_loss / (i + 1)}")

            optimizer.step()
    torch.save(model.state_dict(), "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\reward_predictor.pt")
    plt.plot(plot)
    plt.show()


def balance_files():
    data_path = "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\train_reward"
    counter_1 = 0
    data_path_1 = []
    counter_2 = 0
    data_path_2 = []
    counter_3 = 0
    data_path_3 = []
    for fname in os.listdir(data_path + "\\"):
        for plik in os.listdir(data_path + "\\" + fname):
            if fname == "now":
                path_to_file = data_path + "\\" + fname + "\\" + plik

                state, reward = pickle.load(open(path_to_file, "rb"))

                if torch.all(torch.eq(reward, torch.Tensor([1, 0, 0]))):
                    counter_1 += 1
                    data_path_1.append(path_to_file)
                elif torch.all(torch.eq(reward, torch.Tensor([0, 1, 0]))):
                    counter_2 += 1
                    data_path_2.append(path_to_file)
                elif torch.all(torch.eq(reward, torch.Tensor([0, 0, 1]))):
                    counter_3 += 1
                    data_path_3.append(path_to_file)
    print(f"{counter_1} {counter_2} {counter_3}")
    lowest_value = min(len(data_path_1), len(data_path_2), len(data_path_3))
    balanced_list = data_path_1[:lowest_value] + data_path_2[:lowest_value] + data_path_3[:lowest_value]

    return data_path_1 + data_path_2+data_path_3


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
    model.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1.pt"))
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_reward = Adam(model.parameters(), lr=0.0005)
    plot = []
    plot_reward = []

    for epoch in range(20):
        model.train()
        running_loss = 0.0

        for i, img in enumerate(data_train_loader):
            img_blur, img_sharp = img
            optimizer.zero_grad()

            img_sharp = img_sharp.float().cuda()
            img_blur = img_blur.float().cuda()
            img_deblur = model(img_blur)

            # loss_reward = BCEWithLogitsLoss()(reward , reward_actual)
            loss = mse_loss(img_deblur, img_sharp)

            running_loss += loss.item()

            plot.append(loss.item())

            loss.backward()

            print(f"loss image {running_loss / (i + 1)}")

            optimizer.step()
            # if i == len(data_train_loader)-1:
            #     last_deblurs = img_deblur
        # plt.plot(plot)
        # plt.show()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1.pt")
    plt.plot(plot_reward)
    plt.show()
    print(f"finished epoch {epoch}")
    input_image = img_blur[0][0].detach().cpu().numpy().squeeze()
    predicted_output_img = img_deblur[0].detach().cpu().numpy().squeeze()
    actual_output = img_sharp[0].detach().cpu().numpy().squeeze()
    plt.imshow(input_image)

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
    data_path2 = "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\train_reward"
    # train,val = DeblurDataset(data_path).get_paths()

    train_gan()
    # validate_gan()
    # balance_files()
    #train_reward_model()
