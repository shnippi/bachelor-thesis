import torch
from torch.utils.data import DataLoader, random_split, Subset
from visualization import *
from helper import *
from attacks import *
from LeNet_plus_plus import LeNet_plus_plus
import Data_manager
from loss import entropic_openset_loss
from metrics import *
from dotenv import load_dotenv

load_dotenv()

# Get cpu or gpu device for training.
device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Hyperparameters
batch_size = 128 if torch.cuda.is_available() else 4
epochs = 10 if torch.cuda.is_available() else 5
learning_rate = 0.01
trainsamples = 5000
testsamples = 1000

# create Datasets
# training_data, test_data = Data_manager.mnist_plus_letter(device)
training_data, test_data = Data_manager.mnist_adversarials(device)
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
def train(dataloader, model, loss_fn, optimizer, eps=0.15, eps_iter=0.1):
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

        # TODO: add adversaries and rauschen (rotate in direction of gradient)

        # X, y = random_perturbation(X, y)
        X, y = PGD_attack(X, y, model, loss_fn, eps, eps_iter)

        pred, feat = model(X)

        # print(pred)
        # print(y)

        loss = loss_fn(pred, y)
        loss.backward()

        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


eps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
eps_iter_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]

# tensor that stores the confidences for different eps (rows) and eps_iter (columns)
eps_tensor = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))


# testing loop
def test(dataloader, model, current_epoch=None, eps=None, eps_iter=None):
    # one nested list for each digit + 1 unknown class
    features = [[], [], [], [], [], [], [], [], [], [], []]

    size = len(dataloader.dataset)
    n_batches = len(dataloader)

    model.eval()
    test_loss, conf, correct = 0, 0, 0
    acc_known = torch.tensor((1, 2))
    with torch.no_grad():  # dont need the backward prop
        # iterating over every batch
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, feat = model(X)
            test_loss += loss_fn(pred, y).item()
            # TODO: check if confidence is correct
            conf += confidence(pred, y)
            acc_known += accuracy_known(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # put the 2dfeatures for every sample in the correct sublist according to their true label(index)
            # -1 --> last sublist
            ylist = y.to("cpu").detach().tolist()
            for i in range(len(y)):
                features[ylist[i]].append(feat.to("cpu").detach().tolist()[i])

    test_loss /= n_batches
    correct /= size
    conf /= n_batches

    # plot the features with #classes
    simplescatter(features, 11)

    # update and plot epsilons
    if eps and eps_iter:
        eps_tensor[current_epoch-1][eps_list.index(eps)][eps_iter_list.index(eps_iter)] = conf.item()

        if current_epoch == epochs:
            epsilon_plot(eps_tensor, eps_list, eps_iter_list)

    # TODO: take accuracy of only the knowns
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Confidence: {conf * 100:>0.1f}%, acc_known: {acc_known[0] / acc_known[1] * 100:>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer)
    #     test(test_dataloader, model)

    for eps in eps_list:
        for eps_iter in eps_iter_list:
            # make a new model every time (otherwise it will stack)
            new_model = LeNet_plus_plus().to(device)

            for t in range(epochs):
                print(f"Epoch {t + 1}, eps: {eps}, eps_iter: {eps_iter}\n-------------------------------")
                train(train_dataloader, new_model, loss_fn, optimizer, eps, eps_iter)
                test(test_dataloader, new_model, t + 1, eps, eps_iter)

    print("Done!")
