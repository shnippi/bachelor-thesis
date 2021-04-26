import torch
import os
from numpy import random
from matplotlib import pyplot as plt
from dotenv import load_dotenv

load_dotenv()

device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"


def random_perturbation(X, y):
    # add random perturbation +[-0.1, 0.1] for every pixel for every sample in batch X
    for idx in range(len(X)):
        X[idx][0] += torch.rand(X[idx][0].shape, device=device) * 0.2 * random.choice([-1, 1])

    y = torch.ones(y.shape, dtype=torch.long, device=device) * -1

    # plt.imshow(X[0][0].to("cpu"), "gray")
    # plt.show()

    return X, y
