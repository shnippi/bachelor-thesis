import os, sys
import torch
from matplotlib import pyplot as plt
from numpy import random

device = "cuda:5" if torch.cuda.is_available() else "cpu"


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def random_perturbation(X, y):
    # add random perturbation +[-0.1, 0.1] for every pixel for every sample in batch X
    for idx in range(len(X)):
        X[idx][0] += torch.rand(X[idx][0].shape, device=device) * 0.2 * random.choice([-1, 1])

    y = torch.ones(y.shape, dtype=torch.long, device=device) * -1

    # plt.imshow(X[0][0].to("cpu"), "gray")
    # plt.show()

    return X, y
