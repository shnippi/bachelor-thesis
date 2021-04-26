import torch
from numpy import random
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
from helper import *
from LeNet_plus_plus import LeNet_plus_plus
import Data_manager
from loss import entropic_openset_loss
from metrics import *
from advertorch.attacks import PGDAttack

# Get cpu or gpu device for training.
device = "cuda:5" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Hyperparameters
batch_size = 128 if torch.cuda.is_available() else 4
epochs = 500 if torch.cuda.is_available() else 5
learning_rate = 0.01
trainsamples = 5000
testsamples = 1000

# create Datasets
training_data, test_data = Data_manager.mnist_plus_letter(device, trainsamples, testsamples)
# training_data, test_data = Data_manager.mnist_adversarials(device)
# training_data, test_data = Data_manager.Concat_emnist(device)
# training_data, test_data = Data_manager.mnist_vanilla(device)
# training_data, test_data = Data_manager.emnist_digits(device)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

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
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training loop
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)

    # enumerates the image in greyscale value (X) with the true label (y) in lists that are as long as the batchsize
    # ( 0 (batchnumber) , ( tensor([.. grayscale values ..]) , tensor([.. labels ..]) )  )  <-- for batchsize=1
    for batch, (X, y) in enumerate(dataloader):
        # print(list(enumerate(dataloader))[1]) #prints a batch

        optimizer.zero_grad()

        # if -1 not in y:
        #     continue

        X, y = X.to(device), y.to(device)

        # implicitly calls forward
        pred, feat = model(X)

        # print(pred)
        # print(y)

        loss = loss_fn(pred, y)
        # print(loss)

        # Backpropagation
        loss.backward()

        # plt.imshow(X[0][0].to("cpu"), "gray")
        # plt.show()

        # TODO: add rauschen here (with pred and loss again of the new sample) like training twice in the loop
        # TODO: idea : multiply scalar
        # TODO: idea : add scalar
        # TODO: idea : rotate by a small angle in the direction of gradient (goodfellow)

        # X, y = random_perturbation(X, y)

        # adversary = PGDAttack(
        #     model, loss_fn=loss_fn, eps=0.15,
        #     nb_iter=1, eps_iter=0.1, rand_init=True, clip_min=0.0, clip_max=1.0,
        #     targeted=False)
        #
        # X = adversary.perturb(X,y)
        # y = torch.ones(y.shape, dtype=torch.long, device=device) * -1
        #
        # # plt.imshow(X[0][0].to("cpu"), "gray")
        # # plt.show()
        #
        # pred, feat = model(X)
        #
        # # print(pred)
        # # print(y)
        #
        # loss = loss_fn(pred, y)
        # loss.backward()

        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# testing loop
def test(dataloader, model):
    # one nested list for each digit + 1 unknown class
    features = [[], [], [], [], [], [], [], [], [], [], []]

    size = len(dataloader.dataset)
    n_batches = len(dataloader)

    model.eval()
    test_loss, conf, correct = 0, 0, 0
    with torch.no_grad():  # dont need the backward prop
        # iterating over every batch
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, feat = model(X)
            test_loss += loss_fn(pred, y).item()
            # TODO: check if confidence is correct
            conf += confidence(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # put the 2dfeatures for every sample in the correct sublist according to their true label(index)
            # -1 --> last sublist
            ylist = y.to("cpu").detach().tolist()
            for i in range(len(y) - 1):
                features[ylist[i]].append(feat.to("cpu").detach().tolist()[i])

    # plot the features with #classes
    simplescatter(features, 11)

    test_loss /= n_batches
    correct /= size
    conf /= n_batches
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Confidence: {conf * 100:>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)

    print("Done!")
