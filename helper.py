import os, sys
import torch
from dotenv import load_dotenv
import torch.nn.functional as F
from matplotlib import pyplot as plt

load_dotenv()

device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Only take adversarials if prediction is correct
def filter_correct(X, y, pred):
    correct = torch.argmax(pred, dim=1) == y
    return X[correct], y[correct], y


# Only take adversarials if prediction is over a certain threshold
def filter_threshold(X, y, pred, thresh):
    correct = torch.diagonal(torch.index_select(F.softmax(pred, dim=1), dim=1, index=y) > thresh)
    return X[correct], y[correct], y


# displays a sample
def show_sample(X):
    plt.imshow(X[0][0].to("cpu"), "gray")
    plt.show()
