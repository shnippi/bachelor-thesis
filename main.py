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

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Hyperparameters
batch_size = 64 if torch.cuda.is_available() else 5
epochs = 30 if torch.cuda.is_available() else 1
learning_rate = 1e-3 * 5
trainsamples = 40000
testsamples = 10000


# Download training data from open datasets.
letters_train = datasets.EMNIST(
    root="data",
    split="letters",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
letters_test = datasets.EMNIST(
    root="data",
    split="letters",
    train=False,
    download=True,
    transform=ToTensor(),
)

digits_train = datasets.EMNIST(
    root="data",
    split="digits",
    train=True,
    download=True,
    transform=ToTensor(),
)

digits_test = datasets.EMNIST(
    root="data",
    split="digits",
    train=False,
    download=True,
    transform=ToTensor(),
)

#no printing
with HiddenPrints():
    training_data = ConcatDataset([digits_train, letters_train])
    test_data = ConcatDataset([digits_test, letters_test])

training_data = digits_train
test_data = digits_test

# take different sizes of the datasets depending if GPU is available
if device == "cpu":
    subtrain = list(range(1, 5001))
    subtest = list(range(1, 1001))
    training_data = Subset(training_data, subtrain)
    test_data = Subset(test_data, subtest)
else:
    subtrain = list(range(1, len(training_data) + 1, round((len(training_data) + 1) / trainsamples)))
    subtest = list(range(1, len(test_data) + 1, round((len(test_data) + 1) / testsamples)))
    training_data = Subset(training_data, subtrain)
    test_data = Subset(test_data, subtest)



# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

x,y = list(test_dataloader)[len(test_dataloader) -2]
print(x)
print(y)

# see what dimensions the input is
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


# Define model
model = LeNet_plus_plus().to(device)


class entropic_openset_loss():
    def __init__(self, num_of_classes=10):
        self.num_of_classes = num_of_classes
        self.eye = torch.eye(self.num_of_classes).to(device)
        self.ones = torch.ones(self.num_of_classes).to(device)
        self.unknowns_multiplier = 1. / self.num_of_classes

    def __call__(self, logit_values, target, sample_weights=None):
        # logit_values --> tensor with #batchsize samples, per sample 10 values with logits for each class.
        # target f.e. tensor([0, 4, 1, 9, 2]) with len = batchsize

        # print(logit_values)

        catagorical_targets = torch.zeros(logit_values.shape).to(
            device)  # tensor with size (batchsize, #classes), all logits to 0
        known_indexes = target != -1  # list of bools for the known classes
        unknown_indexes = ~known_indexes  # list of bools for the unknown classes
        catagorical_targets[known_indexes, :] = self.eye[
            target[known_indexes]]  # puts the logits to 1 at the correct index (class) for each sample
        # print(catagorical_targets)
        catagorical_targets[unknown_indexes, :] = self.ones.expand(
            (torch.sum(unknown_indexes).item(), self.num_of_classes)) * self.unknowns_multiplier
        # print(catagorical_targets)

        log_values = F.log_softmax(logit_values, dim=1)  # EOS --> -log(Softmax(x))
        negative_log_values = -1 * log_values
        loss = negative_log_values * catagorical_targets
        # why is there a mean here? --> doesnt matter, leave it. just pump up learning rate
        sample_loss = torch.mean(loss, dim=1)
        # sample_loss = torch.max(loss, dim=1).values
        # print(sample_loss)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss.mean()


loss_fn = entropic_openset_loss()
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.Softmax()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # enumerates the image in greyscale value (X) with the true label (y) in lists that are as long as the batchsize
    # ( 0 (batchnumber) , ( tensor([.. grayscale values ..]) , tensor([.. labels ..]) )  )  <-- for batchsize=1
    for batch, (X, y) in enumerate(dataloader):

        # print(list(enumerate(dataloader))[1]) #prints a batch
        X, y = X.to(device), y.to(device)

        # implicitely calles forward
        pred, feat = model(X)

        loss = loss_fn(pred, y)

        # print(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    # one nested list for each digit
    features = [[], [], [], [], [], [], [], [], [], []]

    size = len(dataloader.dataset)
    print(size)
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

            # for every prediction put the 2dfeatures in the correct sublist according to their true label(index)
            for i in range(len(y) - 1):
                features[ylist[i]].append(feat.to("cpu").detach().tolist()[i])

    simplescatter(features)

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)

    print("Done!")
