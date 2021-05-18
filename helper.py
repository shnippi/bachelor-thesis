import os, sys
import torch
from dotenv import load_dotenv
import torch.nn.functional as F

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
    correct = torch.zeros_like(y, dtype=torch.bool)
    for i in range(len(y)):
        if torch.argmax(pred[i]) == y[i]:
            correct[i] = True

    return X[correct], y[correct]


# Only take adversarials if prediction is over a certain threshold
def filter_threshold(X, y, pred):
    correct = torch.zeros_like(y, dtype=torch.bool)
    for i in range(len(y)):
        if F.softmax(pred[i], dim=0)[y[i]] > 0.5:
            correct[i] = True

    return X[correct], y[correct]
