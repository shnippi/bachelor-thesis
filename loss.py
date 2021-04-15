import torch
from torch.nn import functional as F


class entropic_openset_loss():
    def __init__(self, num_of_classes=10):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_of_classes = num_of_classes
        self.eye = torch.eye(self.num_of_classes).to(self.device)
        self.ones = torch.ones(self.num_of_classes).to(self.device)
        self.unknowns_multiplier = 1. / self.num_of_classes

    def __call__(self, logit_values, target, sample_weights=None):
        # logit_values --> tensor with #batchsize samples, per sample 10 values with logits for each class.
        # target f.e. tensor([0, 4, 1, 9, 2]) with len = batchsize

        # print(logit_values)

        catagorical_targets = torch.zeros(logit_values.shape).to(
            self.device)  # tensor with size (batchsize, #classes), all logits to 0
        known_indexes = target != -1  # list of bools for the known classes
        unknown_indexes = ~known_indexes  # list of bools for the unknown classes
        # print(known_indexes)
        # print(unknown_indexes)
        catagorical_targets[known_indexes, :] = self.eye[
            target[known_indexes]]  # puts the logits to 1 at the correct index (class) for each known sample
        # print(catagorical_targets)
        catagorical_targets[unknown_indexes, :] = self.ones.expand(
            # puts 1/#classes (0.1) for every logit(max entropy)
            (torch.sum(unknown_indexes).item(), self.num_of_classes)) * self.unknowns_multiplier
        # print(catagorical_targets)

        log_values = F.log_softmax(logit_values, dim=1)  # EOS --> -log(Softmax(x))
        # print(log_values)
        negative_log_values = -1 * log_values
        loss = negative_log_values * catagorical_targets
        # print(loss)
        # why is there a mean here? --> doesnt matter, leave it. just pump up learning rate
        sample_loss = torch.mean(loss, dim=1)
        # sample_loss = torch.max(loss, dim=1).values
        # print(sample_loss)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss.mean()
