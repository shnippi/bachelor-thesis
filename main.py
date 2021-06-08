import os
import pathlib
import torch
from torch.utils.data import DataLoader, random_split, Subset
from visualization import *
from helper import *
from attacks import *
from LeNet_plus_plus import LeNet_plus_plus
import Data_manager
import pathlib
from loss import entropic_openset_loss
from metrics import *
from dotenv import load_dotenv
from evaluation import evaluate
# from sklearn.metrics import roc_auc_score
from lots import lots, lots_

load_dotenv()

# Get cpu or gpu device for training.
device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"
metric = os.environ.get('METRIC')
results_dir = pathlib.Path("models")
print("Using {} device".format(device))
print(f"plot: {os.environ.get('PLOT')}, adversary = {os.environ.get('ADVERSARY')}, metric: {metric}")

# Hyperparameters
batch_size = 128 if torch.cuda.is_available() else 4
epochs = 100 if torch.cuda.is_available() else 2
iterations = 3
learning_rate = 0.01
trainsamples = 5000
testsamples = 500

# TODO: MAYBE TRANSFROM THE LETTERS SOMEHOW SO BATCHNORM WORKS? (divide it by mean/std of sth idk)

# create Datasets
# training_data, test_data = Data_manager.mnist_plus_letter(device)
training_data, test_data = Data_manager.mnist_adversarials(device)
# training_data, test_data = Data_manager.Concat_emnist(device)
# training_data, test_data = Data_manager.mnist_vanilla(device)
# training_data, test_data = Data_manager.emnist_digits(device)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)

# see what dimensions the input is
# for X, y in test_dataloader:
#     print("Shape of X [N, C, H, W]: ", X.shape)
#     print("Shape of y: ", y.shape, y.dtype)
#     break

# Define model
model = LeNet_plus_plus().to(device)
if os.environ.get('LOAD') == "t":
    model.load("models/test.model")

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
        # print(list(enumerate(dataloader))[1]) # prints a batch

        optimizer.zero_grad()

        # if -1 not in y:
        #     continue

        X, y = X.to(device), y.to(device)

        # implicitly calls forward
        pred, feat = model(X, features=True)
        # print(feat)
        # print(pred)
        # print(y)

        loss = loss_fn(pred, y)
        # print(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # generate and train on adversaries
        if os.environ.get('ADVERSARY') == "t":
            feat = feat.detach()
            # TODO: find best threshold
            # TODO: detach everything?
            # filter the samples
            # X, y, y_old = filter_correct(X, y, pred)
            X, y, y_old = filter_threshold(X, y, pred, thresh=0.9)

            if len(X) > 0:
                # X, y = random_perturbation(X, y)
                # X, y = PGD_attack(X, y, model, loss_fn, eps, eps_iter)
                # X, y = FGSM_attack(X, y, model, loss_fn)
                # X, y = CnW_attack(X, y, model, loss_fn)
                X, y = lots_attack_batch(X, y, model, feat, y_old, eps)

                pred = model(X)

                # print(pred)
                # print(y)

                loss = loss_fn(pred, y)
                loss.backward()

                optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# eps is upper bound for change of pixel values , educated guess : [0.1:0.5]
eps_list = [0.2, 0.3, 0.4]
eps_iter_list = eps_list
# tensor to store the metric values
eps_tensor = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))
accumulated_eps_tensor = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))


# TODO: clean this mess up
# testing loop
def test(dataloader, model, current_iteration=None, current_epoch=None, eps=None, eps_iter=None):
    model.eval()
    # one nested list for each digit + 1 unknown class
    features = [[], [], [], [], [], [], [], [], [], [], []]
    full_y = []
    full_features = []
    roc_y = torch.tensor([], dtype=torch.long).to(device)
    roc_pred = torch.tensor([], dtype=torch.long).to(device)

    size = len(dataloader.dataset)
    n_batches = len(dataloader)

    test_loss, conf, roc_score, correct = 0, 0, 0, 0
    acc_known = torch.tensor((1, 2))
    with torch.no_grad():  # dont need the backward prop
        # iterating over every batch
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, feat = model(X, features=True)
            full_y.extend(y)
            full_features.extend(feat.tolist())
            test_loss += loss_fn(pred, y).item()
            conf += confidence(pred, y)
            acc_known += accuracy_known(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # add for roc
            roc_y = torch.cat((roc_y, y.detach()))
            roc_pred = torch.cat((roc_pred, pred.detach()))

            # put the 2dfeatures for every sample in the correct sublist according to their true label(index)
            # -1 --> last sublist
            ylist = y.to("cpu").detach().tolist()
            for i in range(len(y)):
                features[ylist[i]].append(feat.to("cpu").detach().tolist()[i])

    test_loss /= n_batches
    correct /= size
    conf /= n_batches
    roc_score = roc(roc_pred.to("cpu").detach(), roc_y.to("cpu").detach())

    # plot the features with #classes
    simplescatter(features, 11)

    # store conf, update and plot epsilons if given
    if eps and eps_iter:
        if metric == "conf":
            eps_tensor[current_epoch - 1][eps_list.index(eps)][eps_iter_list.index(eps_iter)] = conf.item()
        elif metric == "roc":
            eps_tensor[current_epoch - 1][eps_list.index(eps)][eps_iter_list.index(eps_iter)] = roc_score

        if current_epoch == epochs:  # only plot on the last epoch
            epsilon_plot(eps_tensor, eps_list, eps_iter_list, metric, current_iteration)
            epsilon_table(eps_tensor, eps_list, eps_iter_list, metric, current_iteration)
            simplescatter(features, 11, eps, eps_iter, current_iteration)

            # safe model at the end of the iteration
            save_dir = results_dir / f"{eps}_eps_{eps_iter}_epsiter_{current_iteration}iter.model_end"
            results_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir)

            # evaluate results
            evaluate(eps, eps_iter, current_iteration)

    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(
        f"Test Error: \n Confidence: {conf * 100:>0.1f}%, AUC: {roc_score:>0.8f}, "
        f"acc_known: {acc_known[0] / acc_known[1] * 100:>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n")


# TODO: save the flower of the max/ make better flower save system
if __name__ == '__main__':
    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer)
    #     test(test_dataloader, model)

    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer, eps=0.3)
    #     test(test_dataloader, model, 1, t + 1, 0.3, 0.3)

    # TODO: RUN THIS ON MULTIPLE GPU
    for iteration in range(iterations):

        # reset the epsilon tensor
        eps_tensor = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))

        for eps in eps_list:
            for eps_iter in eps_iter_list:

                # only if eps = eps_iter
                if eps != eps_iter:
                    continue

                # seed dependent on current iteration
                torch.manual_seed(iteration)
                new_model = LeNet_plus_plus().to(device)
                new_optimizer = torch.optim.SGD(new_model.parameters(), lr=learning_rate, momentum=0.9)

                for t in range(epochs):
                    print(f"Epoch {t + 1} / {epochs}, eps: {eps}, eps_iter: {eps_iter}, "
                          f"iter: {iteration + 1} / {iterations}\n "
                          f"------------------------------------------")
                    train(train_dataloader, new_model, loss_fn, new_optimizer, eps, eps_iter)
                    test(test_dataloader, new_model, iteration + 1, t + 1, eps, eps_iter)

        accumulated_eps_tensor += eps_tensor

    mean_eps_tensor = accumulated_eps_tensor / iterations
    epsilon_plot(mean_eps_tensor, eps_list, eps_iter_list, metric)
    epsilon_table(mean_eps_tensor, eps_list, eps_iter_list, metric)

    print("Done!")
