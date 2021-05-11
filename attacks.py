import torch
import os
from numpy import random
from matplotlib import pyplot as plt
from dotenv import load_dotenv
from advertorch.attacks import PGDAttack, GradientSignAttack, CarliniWagnerL2Attack

load_dotenv()

device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"


# TODO: GAUSSIAN NOISE?
# TODO: ROTATE BY SMALL ANGLE IN DIRECTION OF GRADIENT (GOODFELLOW)


def random_perturbation(X, y):
    # add random perturbation +[-0.1, 0.1] for every pixel for every sample in batch X
    for idx in range(len(X)):
        X[idx][0] += torch.rand(X[idx][0].shape, device=device) * 0.2 * random.choice([-1, 1])

    y = torch.ones(y.shape, dtype=torch.long, device=device) * -1

    # plt.imshow(X[0][0].to("cpu"), "gray")
    # plt.show()

    return X, y


def PGD_attack(X, y, model, loss_fn, eps=0.15, eps_iter=0.1):
    adversary = PGDAttack(
        model, loss_fn=loss_fn, eps=eps,
        nb_iter=1, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

    X = adversary.perturb(X, y)
    y = torch.ones(y.shape, dtype=torch.long, device=device) * -1

    # plt.imshow(X[0][0].to("cpu"), "gray")
    # plt.show()

    return X, y


def FGSM_attack(X, y, model, loss_fn, eps=0.15, eps_iter=0.1):
    adversary = GradientSignAttack(model, loss_fn=loss_fn, eps=eps, targeted=False)

    X = adversary.perturb(X, y)
    y = torch.ones(y.shape, dtype=torch.long, device=device) * -1

    # plt.imshow(X[0][0].to("cpu"), "gray")
    # plt.show()

    return X, y


# TODO: fix this --> the forward function returns a tuple and this messes it up!!
#  set a flag in forward call for features?
def CnW_attack(X, y, model, loss_fn, eps=0.15, eps_iter=0.1, num_classes=10):
    adversary = CarliniWagnerL2Attack(model.forward, num_classes)

    # print(y)
    # y = y.detach().clone().view(-1, 1)
    # print(y)
    # y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    # print(y_one_hot)

    X = adversary.perturb(X,y)
    y = torch.ones(y.shape, dtype=torch.long, device=device) * -1
    print("hello")
    # plt.imshow(X[0][0].to("cpu"), "gray")
    # plt.show()

    return X, y
