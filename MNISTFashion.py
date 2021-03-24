from itertools import product

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import pytorch_lightning as pl

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# see what dimensions the input is
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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
        # turn into 1D representation (1D per batch element)
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)
        # first fully-connected layer to compute 2D feature space
        z = self.fc1(x)
        # second fully-connected layer to compute the logits
        y = self.fc2(z)
        # return both the logits and the deep features
        return y

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")
