import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.nn import functional as F
import numpy as np
from itertools import chain
from visualization import *
from ConcatDataset import ConcatDataset
from HiddenPrints import HiddenPrints
from LeNet_plus_plus import LeNet_plus_plus
import Data_manager
from loss import entropic_openset_loss

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Hyperparameters
batch_size = 64 if torch.cuda.is_available() else 5
epochs = 30 if torch.cuda.is_available() else 1
learning_rate = 1e-2
trainsamples = 60000
testsamples = 10000

# create Datasets
# training_data, test_data = Data_manager.Concat_digit_letter(device)
training_data, test_data = Data_manager.mnist_vanilla(device)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# see what dimensions the input is
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Define model
model = LeNet_plus_plus().to(device)

# loss function
loss_fn = entropic_openset_loss()
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.Softmax()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # enumerates the image in greyscale value (X) with the true label (y) in lists that are as long as the batchsize
    # ( 0 (batchnumber) , ( tensor([.. grayscale values ..]) , tensor([.. labels ..]) )  )  <-- for batchsize=1
    for batch, (X, y) in enumerate(dataloader):

        # if -1 not in y:
        #     continue

        # print(list(enumerate(dataloader))[1]) #prints a batch
        X, y = X.to(device), y.to(device)

        # implicitely calles forward
        pred, feat = model(X)

        # print(pred)
        # print(y)

        loss = loss_fn(pred, y)

        # print(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# testing loop
def test(dataloader, model):
    # one nested list for each digit + 1 unknown class
    features = [[], [], [], [], [], [], [], [], [], [], []]

    size = len(dataloader.dataset)
    # print(size)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():  # dont need the backward prop
        # iterating over every batch
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, feat = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            ylist = y.to("cpu").detach().tolist()

            # put the 2dfeatures in the correct sublist according to their true label(index). -1 --> last sublist
            for i in range(len(y) - 1):
                features[ylist[i]].append(feat.to("cpu").detach().tolist()[i])

    # plot the features with #classes
    simplescatter(features, 10)

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)

    print("Done!")
