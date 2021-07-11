import os

from torch.utils.data import DataLoader
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

load_dotenv()

# TODO: clean up code (comment toggle etc)
# TODO: get some images from adversarials
# TODO: find best threshold
# TODO: clean up testing loop a bit
# TODO: make a helper func to load all env variables?
# TODO: save the flower of the max/ make better flower save system
# TODO: layernorm? IN THE PIX2PIX THEY USED INSTANCE NORM!!!!!!!!!!!!!! --> try this!!
# TODO: in conv2d layers bias=False because we use batchnorm?
# TODO: declining LR
# TODO: AUC STILL VERY HIGH EVEN WITH NO ADV TRAINING?
# TODO: change the name of the roc plot to auc plot

# Get device and env specifics
device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"
dataset = os.environ.get('DATASET')
filters = os.environ.get('FILTER')
adversary = os.environ.get('ADVERSARY')
results_dir = pathlib.Path("models")
print("Using {} device".format(device))
print(f"adversary = {adversary}, dataset = {dataset}, filter = {filters}"
      f"plot: {os.environ.get('PLOT')}")

# Hyperparameters
batch_size = 128 if torch.cuda.is_available() else 4
epochs = 100 if torch.cuda.is_available() else 1
iterations = 3
learning_rate = 0.01
filter_thresh = 0.9
eps_list = [0.1, 0.2, 0.3, 0.4, 0.5]  # eps is upper bound for change of pixel values , educated guess : [0.1:0.5]
eps_iter_list = eps_list
trainsamples = 5000
testsamples = 1000

# create Datasets
if dataset == "mnist":
    training_data, test_data = Data_manager.mnist(device)
elif dataset == "emnist":
    training_data, test_data = Data_manager.emnist_digits(device)
elif dataset == "mnistletter":
    training_data, test_data = Data_manager.mnist_plus_letter(device)
elif dataset == "emnistconcat":
    training_data, test_data = Data_manager.concat_emnist(device)
else:
    training_data, test_data = Data_manager.open_set(device)

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)

# Define model
model = LeNet_plus_plus().to(device)

# loss function
loss_fn = entropic_openset_loss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# tensors to store the metrics for every eps-epsiter pair at every epoch
eps_tensor_conf = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))
eps_tensor_auc = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))
accumulated_eps_tensor_conf = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))
accumulated_eps_tensor_auc = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))

# list for OSCR curve
eps_oscr_list = []


# training loop
def train(dataloader, model, loss_fn, optimizer, eps=0.15, eps_iter=0.1):
    model.train()
    size = len(dataloader.dataset)

    # enumerates the image in greyscale value (X) with the true label (y) in lists that are as long as the batchsize
    # ( 0 (batchnumber) , ( tensor([.. grayscale values ..]) , tensor([.. labels ..]) )  )  <-- for batchsize=1
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()

        # print(list(enumerate(dataloader))[1]) # prints a batch

        X, y = X.to(device), y.to(device)
        pred, feat = model(X, features=True)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # generate and train adversaries
        if not adversary == "f":
            feat = feat.detach()

            # filter the samples
            if filters == "corr" or filters == "both":
                X, y, y_old = filter_correct(X, y, pred)
            if filters == "thresh" or filters == "both":
                X, y, y_old = filter_threshold(X, y, pred, thresh=filter_thresh)
            else:
                y_old = y

            # check if some samples survived the filter and choose the adversary
            if len(X) > 0:

                if adversary == "rand":
                    X, y = random_perturbation(X, y)
                elif adversary == "pgd":
                    X, y = PGD_attack(X, y, model, loss_fn, eps, eps_iter)
                elif adversary == "fgsm":
                    X, y = FGSM_attack(X, y, model, loss_fn)
                elif adversary == "cnw":
                    X, y = CnW_attack(X, y, model, loss_fn)
                elif adversary == "lots":
                    X, y = lots_attack_batch(X, y, model, feat, y_old, eps)

                pred = model(X)

                loss = loss_fn(pred, y)
                loss.backward()

                optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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

    with torch.no_grad():
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
            roc_pred = torch.cat((roc_pred, torch.nn.functional.softmax(pred, dim=1).detach()))

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

    # TODO: maybe remove this first if
    # store metric, update and plot epsilons if given
    if eps and eps_iter:

        eps_tensor_conf[current_epoch - 1][eps_list.index(eps)][eps_iter_list.index(eps_iter)] = conf.item()
        eps_tensor_auc[current_epoch - 1][eps_list.index(eps)][eps_iter_list.index(eps_iter)] = roc_score

        if current_epoch == epochs:  # only plot on the last epoch
            epsilon_plot(eps_tensor_conf, eps_list, eps_iter_list, "confidence", current_iteration)
            epsilon_plot(eps_tensor_auc, eps_list, eps_iter_list, "Area Under the Curve", current_iteration)
            epsilon_table(eps_tensor_conf, eps_list, eps_iter_list, "confidence", current_iteration)
            epsilon_table(eps_tensor_auc, eps_list, eps_iter_list, "Area Under the Curve", current_iteration)
            # simplescatter(features, 11, eps, eps_iter, current_iteration)

            # safe model at the end of the iteration
            save_dir = results_dir / f"{eps}_eps_{eps_iter}_epsiter_{current_iteration}iter.model_end"
            results_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir)

            # evaluate results
            evaluate(eps, eps_iter, current_iteration)

            add_OSCR(str(eps), eps_oscr_list)

    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(
        f"Test Error: \n Confidence: {conf * 100:>0.1f}%, AUC: {roc_score:>0.8f}, "
        f"acc_known: {acc_known[0] / acc_known[1] * 100:>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer)
    #     test(test_dataloader, model)

    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer, eps=0.3)
    #     test(test_dataloader, model, 1, t + 1, 0.3, 0.3)

    for iteration in range(iterations):

        # reset the epsilon tensor
        eps_tensor_conf = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))
        eps_tensor_auc = torch.zeros((epochs, len(eps_list), len(eps_iter_list)))

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

        accumulated_eps_tensor_conf += eps_tensor_conf
        accumulated_eps_tensor_auc += eps_tensor_auc

        plot_OSCR(eps_oscr_list, "oscr_iter" + str(iteration))
        eps_oscr_list = []

    mean_eps_tensor_conf = accumulated_eps_tensor_conf / iterations
    mean_eps_tensor_auc = accumulated_eps_tensor_auc / iterations
    epsilon_plot(mean_eps_tensor_conf, eps_list, eps_iter_list, "confidence")
    epsilon_plot(mean_eps_tensor_auc, eps_list, eps_iter_list, "Area Under the Curve")
    epsilon_table(mean_eps_tensor_conf, eps_list, eps_iter_list, "confidence")
    epsilon_table(mean_eps_tensor_auc, eps_list, eps_iter_list, "Area Under the Curve")

    print("Done!")
