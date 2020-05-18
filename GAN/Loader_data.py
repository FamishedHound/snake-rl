import pickle
import random

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
from GAN.discriminator import DiscriminatorSmall
from GAN.reward_model import reward_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
Counter = 0


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
        self.counter = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not self.val:
            path = self.dataset[
                index]

            (past, state, np_reward) = pickle.load(open(path, "rb"))

            past_state = torch.cat([past.unsqueeze(0), state.unsqueeze(0)])
            # action_vec = np.zeros(4)
            # action_vec[action] = 1
            # action = action_vec
            # action = torch.ones_like(torch.from_numpy(img)).repeat(4, 1, 1) * torch.from_numpy(action) \
            #     .unsqueeze(1) \
            #     .unsqueeze(2)
            # past_with_action =
            return past,state, np_reward


class RewardPathsDataset(object):
    def __init__(self, data_path):
        self.data_val = []
        self.data_train = []
        self.data_path = data_path

    def get_paths(self):

        for folder in os.listdir(osp.join(self.data_path)):
            for fname in os.listdir(self.data_path + "\\" + folder):
                if folder == "now":
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

    plot = []
    # for shuffle in range(5):
    data_train = RewardDataset(
        RewardPathsDataset("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\train_reward").get_paths().data_train,
        transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=16)

    model = reward_model(6).cuda()

    # model.load_state_dict(
    #     torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\reward_predictor_future_2frame.pt"))

    model.train()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(21):

        running_loss = 0.0

        for i, img in enumerate(data_train_loader):
            state, reward = img
            reward = reward.float()
            optimizer.zero_grad()

            actual_reward = reward.cuda()
            state = state.float().cuda()
            models_reward = model(state)
            actual_reward = torch.argmax(actual_reward, dim=1)
            loss_reward = CrossEntropyLoss()(models_reward, actual_reward.long())
            # if actual_reward.item()==0:
            #     plt.imshow(state.cpu().numpy().squeeze()[0],cmap='gray',vmin=0,vmax=1)
            #     plt.show()
            #     plt.imshow(state.cpu().numpy().squeeze()[1],cmap='gray',vmin=0,vmax=1)
            #     plt.show()
            #     print()

            running_loss += loss_reward.item()

            plot.append(loss_reward.item())

            loss_reward.backward()
            print("predicted reward {} actual reward {}".format(torch.argmax(models_reward[0]).item(),
                                                                actual_reward[0].item()))
            print(f"loss image {running_loss / (i + 1)}")
            optimizer.step()
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\reward_predictor_future_2frame_new.pt")

    # print("finished shuffle {} ".format(shuffle))
    torch.save(model.state_dict(),
               "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\reward_predictor_future_2frame_new.pt")
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
    counter = 0

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
                counter += 1

    print(f"{counter_1} {counter_2} {counter_3}")
    lowest_value = min(len(data_path_1), len(data_path_2), len(data_path_3))
    balanced_list = random.sample(data_path_1, lowest_value) + random.sample(data_path_2, lowest_value) + random.sample(
        data_path_3, 3 * lowest_value)

    return data_path_1 + data_path_2 + data_path_3


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
    data_train = RewardDataset(
        RewardPathsDataset("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\train_reward").get_paths().data_train,
        transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=2)
    model = UNet(6, 2).cuda()

    # model.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1_2frame.pt"))
    discriminator = DiscriminatorSmall(1).cuda()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_discrimnator = SGD(discriminator.parameters(), lr=0.01, momentum=0.9)

    gan_loss = torch.nn.BCELoss().cuda()
    plot = []
    plot_reward = []

    for epoch in range(1):
        model.train()
        running_loss = 0.0

        for i, img in enumerate(data_train_loader):
            current_state, future, reward, action = img
            optimizer.zero_grad()
            #Generator
            img_sharp = future.cuda()
            img_blur = current_state.cuda()
            img_deblur = model(img_blur)

            # loss_reward = BCEWithLogitsLoss()(reward , reward_actual)
            loss_mse = mse_loss(target=img_deblur, input=img_sharp)
            loss_gan = gan_loss(img_deblur, torch.ones_like(img_deblur))
            running_loss += loss_mse.item()

            plot.append(loss_mse.item())
            loss_gan_mse = loss_gan+loss_mse
            loss_gan_mse.backward()
            optimizer.step()
            #Discriminator
            optimizer_discrimnator.zero_grad()
            disc_true = discriminator(img_sharp)
            disc_fake = discriminator(img_deblur)
            disc_true_loss =gan_loss(disc_true, torch.zeros_like(disc_true))
            disc_fake_loss =gan_loss(disc_fake, torch.zeros_like(disc_fake))

            discriminator_loss = disc_true_loss+disc_fake_loss
            discriminator_loss.backward()
            optimizer_discrimnator.step()

            print(f"loss image {running_loss / (i + 1)}")


            # if i == len(data_train_loader)-1:
            #     last_deblurs = img_deblur
        # plt.plot(plot)
        # plt.show()
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1_2frame_with_discriminator.pt")
    plt.plot(plot_reward)
    plt.show()
    print(f"finished epoch {epoch}")
    input_image = img_blur[0][0].detach().cpu().numpy().squeeze()
    predicted_output_img = img_deblur[0].detach().cpu().numpy().squeeze()
    actual_output = img_sharp[0].detach().cpu().numpy().squeeze()

    plt.imshow(predicted_output_img[0], cmap='gray', vmax=1, vmin=0)
    plt.show()
    plt.imshow(predicted_output_img[1], cmap='gray', vmax=1, vmin=0)
    plt.show()
    plt.imshow(actual_output[0], cmap='gray', vmax=1, vmin=0)
    plt.show()
    plt.imshow(actual_output[1], cmap='gray', vmax=1, vmin=0)
    plt.show()
    torch.save(model.state_dict(), "C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1_2frame_with_discriminator.pt")
    plt.plot(plot)
    plt.show()


def get_accuracy_reward_predictor():
    from torchvision import transforms

    from torch.utils.data import DataLoader

    train_transforms_list = [transforms.ToTensor(),
                             transforms.ToPILImage()]
    train_transforms = transforms.Compose(train_transforms_list)

    data_train = RewardDataset(balance_files(), transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=1, shuffle=True, num_workers=16)
    all = 0
    right = 0

    with torch.no_grad():
        model = reward_model(5).cuda()
        model.load_state_dict(
            torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\reward_predictor.pt"))
        model.eval()
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(1):

            for i, img in enumerate(data_train_loader):
                state, actual_reward = img
                optimizer.zero_grad()

                actual_reward = actual_reward.float().cuda()
                state = state.float().cuda()
                models_reward = model(state)
                actual_reward = torch.argmax(actual_reward, dim=1)
                if torch.argmax(models_reward[0]).item() == actual_reward[0].item():
                    right += 1

                # print("predicted reward {} actual reward {}".format(torch.argmax(models_reward[0]).item(),
                #                                                     actual_reward[0].item()))
                all += 1

    print(f" ACCURACY IS {right / all} FROM WHICH RIGHT {right} AND NO. OF DATA {all}")


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

    model = UNet(6, 2).cuda()
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
    # get_accuracy_reward_predictor()
