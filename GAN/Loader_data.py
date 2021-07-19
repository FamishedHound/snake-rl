import pickle
import numpy as np
import torchvision
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
import traceback

from GAN import discriminator
# from GAN.discriminator import DiscriminatorSmall
from GAN.model import UNet
from torch.utils.data import DataLoader

from GAN.reward_model import reward_model
from GAN.testing_architectures import GeneratorSmall, DiscriminatorSmall

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None, num_images=1, val=False):
        self.dataset = dataset
        self.transform = transform
        self.val = val

    def __len__(self):
        return 140000


    def __getitem__(self, index):
        try :
            if not self.val:
                path = self.dataset[
                    index]  # Remember that S_images has 1 image more than Sa_images because index ffor Sa is index-1
                path_output = self.dataset[index].replace("S_images", "Sa_images")

                pickled_arr, _ = pickle.load(open(path, "rb"))
                pickled_arr_output = pickle.load(open(path_output, "rb"))
                noise = torch.randn(1, 84, 84)
                noise = noise
                # if self.transform is not None:
                #     img = self.transform(img)
                # for experimental self.transform(pickled_arr_output.to(torch.float32))
                return  pickled_arr, pickled_arr_output  # was pickled_arr, pickled_arr_output
        except Exception:
            print("the fuck")
            traceback.print_exc()
            exit(555)

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
                elif folder == "S_images":
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

    for epoch in range(7):
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
            print("predicted reward {} actual reward {}".format(torch.argmax(models_reward[0]).item(),
                                                                actual_reward[0].item()))
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

    return data_path_1 + data_path_2 + data_path_3


def train_gan():
    from torchvision import transforms
    train_transforms_list = [transforms.ToTensor(),
                             # transforms.Normalize(mean, std)
                             ]
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    counter = 0
    train_transforms_list = [transforms.ToTensor(),
                             transforms.ToPILImage()]
    train_transforms = transforms.Compose(train_transforms_list)
    data_train = ImageDataset(DeblurDataset(data_path).get_paths().data_train, transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=32,shuffle=True,num_workers=4) #comeback shuffle=True
    model = UNet(5, 1).cuda()
    #model.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\new_models\\GAN11_new.pt"))
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    discriminator = DiscriminatorSmall(2).cuda()
    #discriminator.load_state_dict(
    #    torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\DISC1235678RIMINATOR3.pt"))
    optimizer_reward = Adam(model.parameters(), lr=3e-4)
    plot = []
    plot_reward = []
    optimizer_discrimnator = SGD(discriminator.parameters(), lr=0.01, momentum=0.9)
    l1_loss = torch.nn.L1Loss().cuda()

    gan_loss = torch.nn.BCELoss().cuda()

    # while x <= 1000:
    #     alpha = x
    generator_amplifier = 3
    discriminator_deamplifier = 15
    for epoch in range(30):
        model.train()
        running_loss = 0.0
        starting_gan = 0.1
        for i, img in enumerate(data_train_loader):
            img_blur, img_sharp = img
            optimizer.zero_grad()

            img_sharp = img_sharp.float().cuda()
            img_blur = img_blur.float().cuda()
            img_deblur = model(img_blur)

            # loss_reward = BCEWithLogitsLoss()(reward , reward_actual)
            loss = l1_loss(img_deblur, img_sharp)
            generator = discriminator(img_deblur)
            loss_gan = gan_loss(generator, torch.ones_like(generator))
            running_loss += loss.item()

            plot.append(loss.item())
            dis_mse_loss = loss*generator_amplifier + loss_gan/discriminator_deamplifier
            # if dis_mse_loss > 1:
            #     starting_gan*=100
            # if dis_mse_loss < 0.1:
            #     starting_gan*=1.2

            dis_mse_loss.backward()

            # print(f"loss image {running_loss / (i + 1)} for alpha {alpha}")
            print(f"combined Loss : {dis_mse_loss} with starting gan being {starting_gan}")
            optimizer.step()

            optimizer_discrimnator.zero_grad()
            disc_true = discriminator(img_sharp)
            disc_fake = discriminator(img_deblur.detach())
            disc_true_loss = gan_loss(disc_true, torch.ones_like(disc_true))
            disc_fake_loss = gan_loss(disc_fake, torch.zeros_like(disc_fake))

            discriminator_loss = disc_true_loss + disc_fake_loss
            discriminator_loss.backward()
            optimizer_discrimnator.step()
            # if i == len(data_train_loader)-1:
            #     last_deblurs = img_deblur
        print(f"finished epoch {epoch}")
        # plt.plot(plot)
        # plt.show()
        if epoch % 2 == 0:
            torch.save(discriminator.state_dict(),
                       f"C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\new_models\\discriminator13_{generator_amplifier}_{discriminator_deamplifier}_new.pt")
            torch.save(model.state_dict(),
                       f"C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\new_models\\GAN13_{generator_amplifier}_{discriminator_deamplifier}_new.pt")
        plt.plot(plot_reward)
        plt.show()

        input_image = img_blur[0][0].detach().cpu().numpy().squeeze()
        predicted_output_img = img_deblur[0].detach().cpu().numpy().squeeze()
        actual_output = img_sharp[0].detach().cpu().numpy().squeeze()
        plt.imshow(input_image, cmap='gray', vmin=0, vmax=1)
        save_plot_and_dump_pickle(counter, input_image,"input")
        plt.show()
        counter+=1
        plt.imshow(predicted_output_img, cmap='gray', vmin=0, vmax=1)
        plt.savefig(f'C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\images\\{counter}_gan_response', bbox_inches='tight')
        plt.show()
        counter += 1
        plt.imshow(actual_output, cmap='gray', vmin=0, vmax=1)
        plt.savefig(f'C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\images\\{counter}_ground_truth',
                    bbox_inches='tight')
        plt.show()
        counter += 1
        torch.save(discriminator.state_dict(),
                   f"C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\new_models\\discriminator13_{generator_amplifier}_{discriminator_deamplifier}_new.pt")
        torch.save(model.state_dict(),
                   f"C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\new_models\\GAN13_{generator_amplifier}_{discriminator_deamplifier}_new.pt")
        plt.plot(plot)
        plt.show()
        # x*=10


def save_plot_and_dump_pickle(counter, input_image,source):
    plt.savefig(f'C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\images\\{counter}_input', bbox_inches='tight')
    with open(f"C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\images\\{counter}_{source}.pickle", 'wb') as handle:
        pickle.dump(input_image, handle, protocol=pickle.HIGHEST_PROTOCOL)


def validate_prepare_data():
    pickled_arr = pickle.load(
        open("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\validate_gan\\state_s_188.pickle", "rb"))
    # pickled_arr_output = pickle.load(
    #    open("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\validate_gan\\state_s_1.pickle", "rb"))

    return (pickled_arr, None)


def experimental_train():
    train_transforms_list = [transforms.ToPILImage(),
                             transforms.Resize((20, 20)),
                             transforms.ToTensor()]
    train_transforms = transforms.Compose(train_transforms_list)

    data_train = ImageDataset(DeblurDataset(data_path).get_paths().data_train, transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=8)

    discriminator = DiscriminatorSmall(32).cuda()
    model = GeneratorSmall(32).cuda()

    optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optim_generator = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.999))
    crit_discriminator = torch.nn.BCELoss().cuda()
    crit_generator = torch.nn.BCELoss().cuda()
    for _ in range(8):
        model.train()
        discriminator.train()
        for i, img in enumerate(data_train_loader):
            img_blur, img_sharp = img

            img_sharp = img_sharp.float().cuda()
            img_blur = img_blur.float().cuda()

            # Generate noise
            noise = torch.randn(64, 100, 1, 1).cuda()
            fake_imgs = model(noise)

            optim_discriminator.zero_grad()
            # train with real
            pred_true = discriminator(img_sharp)
            loss_disc_true = crit_discriminator(pred_true, torch.ones_like(pred_true))

            # train with fake
            pred_fake = discriminator(fake_imgs.detach())
            loss_disc_fake = crit_discriminator(pred_fake, torch.zeros_like(pred_fake))

            loss_disc = loss_disc_true + loss_disc_fake
            loss_disc.backward()

            optim_discriminator.step()
            print(loss_disc.item())
            # Generator
            optim_generator.zero_grad()

            pred_fake_gen = discriminator(fake_imgs)
            loss_gen = crit_generator(pred_fake_gen, torch.ones_like(pred_fake_gen))
            loss_gen.backward()

            optim_generator.step()

            # second iteration
            optim_generator.zero_grad()
            noise = torch.randn(64, 100, 1, 1).cuda()
            fake_imgs = model(noise)
            pred_fake_gen = discriminator(fake_imgs)
            loss_gen = crit_generator(pred_fake_gen, torch.ones_like(pred_fake_gen))
            loss_gen.backward()

            optim_generator.step()

            # second iteration
            optim_generator.zero_grad()
            noise = torch.randn(64, 100, 1, 1).cuda()
            fake_imgs = model(noise)
            pred_fake_gen = discriminator(fake_imgs)
            loss_gen = crit_generator(pred_fake_gen, torch.ones_like(pred_fake_gen))
            loss_gen.backward()

            optim_generator.step()

            # second iteration
            optim_generator.zero_grad()
            noise = torch.randn(64, 100, 1, 1).cuda()
            fake_imgs = model(noise)
            pred_fake_gen = discriminator(fake_imgs)
            loss_gen = crit_generator(pred_fake_gen, torch.ones_like(pred_fake_gen))
            loss_gen.backward()

            optim_generator.step()

        plt.imshow(fake_imgs[0].detach().cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
        plt.show()
        plt.imshow(fake_imgs[1].detach().cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
        plt.show()
        plt.imshow(fake_imgs[2].detach().cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
        plt.show()
        torch.save(discriminator.state_dict(),
                   f"C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\Experimental_dis.pt")
        torch.save(model.state_dict(),
                   f"C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\Experimental_gen.pt")


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
    # train_reward_model()
    # experimental_train()
