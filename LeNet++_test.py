from itertools import product

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
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

# Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 1e-3

# only taking split
# train_data = datasets.MNIST('data', train=True, download=True, transform=ToTensor())
# train, val = random_split(train_data, [55000, 5000])
# train_dataloader = DataLoader(train, batch_size=32)
# test_dataloader = DataLoader(val, batch_size=32)


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

        #list for featurerepresentation
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

    def __call__(self, logit_values, target, sample_weights=None):
        catagorical_targets = torch.zeros(logit_values.shape).to(device)
        known_indexes = target != -1  # list of bools for the known indexes
        unknown_indexes = ~known_indexes  # list of bools for the unknown indexes
        catagorical_targets[known_indexes, :] = self.eye[target[known_indexes]]
        # print(catagorical_targets)
        catagorical_targets[unknown_indexes, :] = self.ones.expand(
            (torch.sum(unknown_indexes).item(), self.num_of_classes)) * self.unknowns_multiplier
        # print(catagorical_targets)

        log_values = F.log_softmax(logit_values, dim=1)
        negative_log_values = -1 * log_values
        loss = negative_log_values * catagorical_targets
        # TODO: why is there a mean here?
        sample_loss = torch.mean(loss, dim=1)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss.mean()


#loss_fn = entropic_openset_loss()
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.Softmax()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
featurearray = np.array([])


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    features = []
    # enumerates the image in greyscale value (X) with the true label (y) in lists that are as long as the batchsize
    # ( 0 (batchnumber) , tensor([.. grayscale values ..]) , tensor([.. labels ..]) )  <-- for batchsize=1
    for batch, (X, y) in enumerate(dataloader):

        # print(list(enumerate(dataloader))[1]) #prints a batch
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        features.append(model.featurerepr.to("cpu").detach().tolist())

        # print(pred)
        # print(y)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return list(chain.from_iterable(features))


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

    print(len(features))
    plt.scatter(*zip(*features))
    plt.show()

print("Done!")