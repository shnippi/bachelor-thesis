import os
import torch
from torch.nn import functional as F
from dotenv import load_dotenv

load_dotenv()

device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"

"""
implementation taken from https://github.com/Vastlab/vast
"""


class entropic_openset_loss():
    def __init__(self, num_of_classes=10):
        self.device = device
        self.num_of_classes = num_of_classes
        self.eye = torch.eye(self.num_of_classes).to(self.device)
        self.ones = torch.ones(self.num_of_classes).to(self.device)
        self.unknowns_multiplier = 1. / self.num_of_classes

    def __call__(self, logit_values, target, sample_weights=None):
        # logit_values --> tensor with #batchsize samples, per sample 10 values with logits for each class.
        # target f.e. tensor([0, 4, 1, 9, 2]) with len = batchsize

        # check if logit_values is tuple --> model returned x,y (x : preditctions, y : features)
        if isinstance(logit_values, tuple):
            logit_values = logit_values[0]

        categorical_targets = torch.zeros(logit_values.shape).to(
            self.device)  # tensor with size (batchsize, #classes), all logits to 0
        known_indexes = target != -1  # list of bools for the known classes
        unknown_indexes = ~known_indexes  # list of bools for the unknown classes
        categorical_targets[known_indexes, :] = self.eye[
            target[known_indexes]]  # puts the logits to 1 at the correct index (class) for each known sample
        categorical_targets[unknown_indexes, :] = self.ones.expand(
            # puts 1/#classes (0.1) for every logit(max entropy)
            (torch.sum(unknown_indexes).item(), self.num_of_classes)) * self.unknowns_multiplier

        log_values = F.log_softmax(logit_values, dim=1)  # EOS --> -log(Softmax(x))
        negative_log_values = -1 * log_values
        loss = negative_log_values * categorical_targets  # puts the -log-values at index for each sample (rest is 0)
        sample_loss = torch.mean(loss, dim=1)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss.mean()
