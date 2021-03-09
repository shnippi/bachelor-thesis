import torch

from torch import nn

from torch import optim

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

model = nn.Sequential(nn.Linear(28 * 28, 64),
                      nn.ReLU(),
                      nn.Linear(64, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10)
                      )

if torch.cuda.is_available():
    model = model.cuda()

optimiser = optim.SGD(model.parameters(), lr=1e-2)

loss = nn.CrossEntropyLoss()

nb_epochs = 5

for epoch in range(nb_epochs):
    losses = list()
    for batch in train_loader:
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)
        if torch.cuda.is_available():
            x = x.cuda()

        l = model(x)  # logit
        J = loss(l, y.cuda() if torch.cuda.is_available() else y)

        model.zero_grad()

        J.backward()

        optimiser.step()

        losses.append(J.item())

    print(f'Epoch {epoch + 1}, train loss: {torch.tensor(losses).mean():.2f}')

    losses = list()
    for batch in val_loader:
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)
        if torch.cuda.is_available():
            x = x.cuda()

        with torch.no_grad():
            l = model(x)  # logit

        J = loss(l, y.cuda() if torch.cuda.is_available() else y)

        losses.append(J.item())

    print(f'Epoch {epoch + 1}, validation loss: {torch.tensor(losses).mean():.2f}')
