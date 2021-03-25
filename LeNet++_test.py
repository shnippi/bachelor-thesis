from itertools import product

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

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Hyperparameters
batch_size = 64 if torch.cuda.is_available() else 5
epochs = 5 if torch.cuda.is_available() else 1
learning_rate = 1e-3

# smaller datasets if no GPU available

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# take subset of training set if no GPU available
if device == "cpu":
    subtrain = list(range(1, 5001))
    subtest = list(range(1, 1001))
    training_data = Subset(training_data, subtrain)
    test_data = Subset(test_data, subtest)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# see what dimensions the input is
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


# Define model
class LeNet_plus_plus(nn.Module):
    def __init__(self):
        super(LeNet_plus_plus, self).__init__()

        # list for featurerepresentation
        self.featurerepr = []

        # first convolution block
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(in_channels=self.conv1_1.out_channels, out_channels=32, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm1 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # second convolution block
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_2.out_channels, out_channels=64, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv2_1.out_channels, out_channels=64, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm2 = nn.BatchNorm2d(self.conv2_2.out_channels)
        # third convolution block
        self.conv3_1 = nn.Conv2d(in_channels=self.conv2_2.out_channels, out_channels=128, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.conv3_2 = nn.Conv2d(in_channels=self.conv3_1.out_channels, out_channels=128, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm3 = nn.BatchNorm2d(self.conv3_2.out_channels)
        # fully-connected layers
        self.fc1 = nn.Linear(in_features=self.conv3_2.out_channels * 3 * 3,
                             out_features=2, bias=True)
        self.fc2 = nn.Linear(in_features=2, out_features=10, bias=True)
        # activation function
        self.prelu_act = nn.PReLU()

    def forward(self, x):
        # compute first convolution block output
        x = self.prelu_act(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        # compute second convolution block output
        x = self.prelu_act(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        # compute third convolution block output
        x = self.prelu_act(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        # flattens it --> turn into 1D representation (1D per batch element)
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)
        # first fully-connected layer to compute 2D feature space. THIS IS THE 2D FEATURE VECTOR SPACE
        self.featurerepr = self.fc1(x)
        # second fully-connected layer to compute the logits
        y = self.fc2(self.featurerepr)
        # return both the logits and the deep features. THIS IS THE PREDICTION
        return y


model = LeNet_plus_plus().to(device)


class entropic_openset_loss():
    def __init__(self, num_of_classes=10):
        self.num_of_classes = num_of_classes
        self.eye = torch.eye(self.num_of_classes).to(device)
        self.ones = torch.ones(self.num_of_classes).to(device)
        self.unknowns_multiplier = 1. / self.num_of_classes

    def __call__(self, logit_values, target, sample_weights=None):  # logit_values --> tensor with #batchsize#
        # samples, per sample 10 values with the respective logits for each class.
        # target f.e. tensor([0, 4, 1, 9, 2]) with len = batchsize

        catagorical_targets = torch.zeros(logit_values.shape).to(
            device)  # tensor with size (batchsize, #classes), all logits to 0
        known_indexes = target != -1  # list of bools for the known indexes
        unknown_indexes = ~known_indexes  # list of bools for the unknown indexes
        catagorical_targets[known_indexes, :] = self.eye[
            target[known_indexes]]  # puts the logits to 1 at the correct index for each sample
        # print(catagorical_targets)
        catagorical_targets[unknown_indexes, :] = self.ones.expand(
            (torch.sum(unknown_indexes).item(), self.num_of_classes)) * self.unknowns_multiplier
        # print(catagorical_targets)

        log_values = F.log_softmax(logit_values, dim=1)  # TODO: why take log? vanishing numbers/gradients?
        # print(log_values)
        negative_log_values = -1 * log_values
        loss = negative_log_values * catagorical_targets
        # TODO: why is there a mean here?
        # print(loss)
        sample_loss = torch.mean(loss, dim=1)
        # print(sample_loss)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss.mean()


# loss_fn = entropic_openset_loss()
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.Softmax()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
featurearray = np.array([])


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # one nested list for each feature
    features = [[], [], [], [], [], [], [], [], [], []]
    # enumerates the image in greyscale value (X) with the true label (y) in lists that are as long as the batchsize
    # ( 0 (batchnumber) , ( tensor([.. grayscale values ..]) , tensor([.. labels ..]) )  )  <-- for batchsize=1
    for batch, (X, y) in enumerate(dataloader):

        # print(list(enumerate(dataloader))[1]) #prints a batch
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        ylist = y.to("cpu").detach().tolist()
        xlist = X.to("cpu").detach().tolist()

        # put the 2dfeatures in the correct sublist according to their true label
        for i in range(len(y) - 1):
            features[ylist[i]].append(model.featurerepr.to("cpu").detach().tolist()[i])

        # print(pred)
        # print(y)
        # print(model.featurerepr)
        # print(features)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return features


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():  # dont need the backward prop
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    features = train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)

    colors = ["b", "g", "r", "c", "m", "y", "k", "lawngreen", "peru", "deeppink"]

    # scatterplot every digit to a color
    for i in range(10):
        plt.scatter(*zip(*(features[i])), c=colors[i])

    plt.show()

print("Done!")
