#%%

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.optim.adam import Adam
import time

#%%

# check if gpu is available, otherwise cpu will be used

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

#%%

# Init parameters
num_threads = torch.get_num_threads()
batch_size = 64
num_epochs = 20
learning_rate = 0.01

#%%

# define a transform to convert to images to tensor and normalize
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,),)]) # mean adn std have t be suqences (e.g.m tuples)


#%%

# Load the data from Fashion MNIST datasets (train and test set)

data = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_data = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))


#%%

# Data loader - train and test



train_set, val_set = torch.utils.data.random_split(data, [50000, 10000])
trainloader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(val_set,
                                          batch_size=batch_size,shuffle=True)

# Define nn model class
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.2)
        self.lin1 = nn.Linear(1152, 128)
        self.lin2 = nn.Linear(128, 10)

    def forward(self, x):
        #x = x.view(x.shape[0], -1)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = x.view(x.shape[0], -1) #Flatten
        x = self.lin1(F.relu(x))
        x = self.drop(x)
        x = self.lin2(F.relu(x))


        return F.log_softmax(x,dim=1)

#%%

# Init model class and move to GPU (CUDA)
from torchsummary import summary

model = Classifier().cuda()


summary(model, (1,28,28))

#%%

# Init loss class - The negative log likelihood loss (NLL Loss)

error = torch.nn.NLLLoss()


#%%

# Init optimizer class - SGD

optimizer = Adam(model.parameters(), lr=3e-4)

#%%

# Set model to training mode

model.train()

#%%

# Train the model - loop
train_losses = []
for epoch in range(num_epochs):
    time_stamp = time.time()
    running_loss = 0

    # Load images, labels
    for i, data in enumerate(trainloader,0):
        # Move images, labels to CUDA
        images, labels = data

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output
        outputs = model(images.cuda())

        # Calculate Loss
        loss = error(outputs.cuda(), labels.cuda())
        #print(loss.item())
        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        running_loss += loss.item()*images.size(0)
    running_loss = running_loss/len(trainloader.sampler)
    print(f"Running lose {running_loss}")

    # running_loss = running_loss / len(trainloader.sampler)
    # train_losses.append(running_loss)
    #
    # print('Epoch: {} \tTraining loss: {:.6f} \tTime:{}'.format(
    #     epoch+1, running_loss, time.time() - time_stamp))

#%%
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        images, labels = data
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
print('Accuracy of the network : %d %%' % (
    100 * correct / total))

