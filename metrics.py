import torch
from dotenv import load_dotenv
import os
from sklearn.metrics import roc_auc_score

load_dotenv()

device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"

"""This file contains different metrics that can be applied to evaluate the training"""


def accuracy_known(prediction, target):
    """Computes the classification accuracy of the classifier based on known samples only.
  Any target that does not belong to a certain class (target is -1) is disregarded.

  Parameters:

    prediction: the output of the network, can be logits or softmax scores

    target: the vector of true classes; can be -1 for unknown samples

  Returns a tensor with two entries:

    correct: The number of correctly classified samples

    total: The total number of considered samples


    implementation taken from https://github.com/Vastlab/vast
  """

    with torch.no_grad():
        known = target >= 0

        total = torch.sum(known, dtype=int)
        if total:
            correct = torch.sum(torch.max(prediction[known], axis=1).indices == target[known], dtype=int)
        else:
            correct = 0

    return torch.tensor((correct, total))


def sphere(representation, target, sphere_radius=None):
    """Computes the radius of unknown samples.
  For known samples, the radius is computed and added only when sphere_radius is not None.

  Parameters:

    representation: the feature vector of the samples

    target: the vector of true classes; can be -1 for unknown samples

  Returns a tensor with two entries:

    length: The sum of the length of the samples

    total: The total number of considered samples


    implementation taken from https://github.com/Vastlab/vast
  """

    with torch.no_grad():
        known = target >= 0

        magnitude = torch.norm(representation, p=2, dim=1)

        sum = torch.sum(magnitude[~known])
        total = torch.sum(~known)

        if sphere_radius is not None:
            sum += torch.sum(torch.clamp(sphere_radius - magnitude, min=0.))
            total += torch.sum(known)

    return torch.tensor((sum, total), device=device)


def confidence(logits, target, negative_offset=0.1):
    """Measures the softmax confidence of the correct class for known samples,
  and 1 + negative_offset - max(confidence) for unknown samples.

  Parameters:

    logits: the output of the network, must be logits

    target: the vector of true classes; can be -1 for unknown samples

  Returns a tensor with two entries:

    confidence: the sum of the confidence values for the samples

    total: The total number of considered samples

    implementation taken from https://github.com/Vastlab/vast
  """

    with torch.no_grad():
        known = target >= 0

        pred = torch.nn.functional.softmax(logits, dim=1)
        #    import ipdb; ipdb.set_trace()

        confidence = 0.
        if torch.sum(known):
            # sums the softmax probabilities at the index of the targets
            confidence += torch.sum(pred[known, target[known]])
        if torch.sum(~known):
            # sum( 1 + negative offset - (the highest softmax probabilities of the guesses) )
            # --> if the highest prob is 0.15 (for 7 f.e.), it is 0.95 confident that its sth else
            confidence += torch.sum(1. + negative_offset - torch.max(pred[~known], dim=1)[0])

        # TODO: divide by the length of logits?

    return confidence / len(logits)


def roc(pred, y):
    scores = torch.ones_like(y, dtype=torch.float)
    target = torch.ones_like(y)

    # binary roc_auc with label 0 for unknowns and label 1 for knowns
    for i in range(len(y)):
        if y[i] == -1:
            scores[i] = torch.max(pred[i]).item()
            target[i] = 0
        else:
            scores[i] = pred[i][y[i]].item()

    scores = scores.detach().numpy()
    target = target.detach().numpy()

    return roc_auc_score(target, scores)


def tensor_OSRC(gt, predicted_class, score):

    """
    implementation taken from https://github.com/Vastlab/vast
    """

    if len(score.shape) != 1:
        score = score[:, 0]
    score = score.to(device)
    score, indices = torch.sort(score, descending=True)
    indices = indices.cpu()
    predicted_class, gt = predicted_class[indices], gt[indices]
    del indices

    # Reverse score order so that the last occurence of the highest threshold is preserved
    scores_reversed = score[torch.arange(score.shape[0] - 1, -1, -1)]
    unique_scores_reversed, counts_reversed = torch.unique_consecutive(scores_reversed, return_counts=True)
    del scores_reversed
    # Reverse again to get scores & counts in descending order
    indx = torch.arange(unique_scores_reversed.shape[0] - 1, -1, -1)
    unique_scores, counts = unique_scores_reversed[indx], counts_reversed[indx]
    del unique_scores_reversed, counts_reversed

    gt = gt.to(device)
    # Get the labels for unknowns
    unknown_labels = set(torch.flatten(gt).tolist()) - set(torch.flatten(predicted_class).tolist())

    # Get all indices for knowns and unknowns
    all_known_indexs = []
    for unknown_label in unknown_labels:
        all_known_indexs.append(gt != unknown_label)
    all_known_indexs = torch.stack(all_known_indexs)
    known_indexs = all_known_indexs.all(dim=0)
    unknown_indexs = ~known_indexs
    del all_known_indexs

    # Get the denominators for accuracy and OSE
    no_of_knowns = known_indexs.sum().type('torch.FloatTensor')
    no_of_unknowns = unknown_indexs.sum().type('torch.FloatTensor')

    all_unknowns = torch.cumsum(unknown_indexs, dim=-1).type('torch.FloatTensor')
    OSE = all_unknowns / no_of_unknowns

    correct = torch.any(gt[:, None].cpu() == predicted_class, dim=1)
    correct = correct.to(device)
    correct = torch.cumsum(correct, dim=-1).type('torch.FloatTensor')

    knowns_accuracy = correct / no_of_knowns
    threshold_indices = torch.cumsum(counts, dim=-1) - 1
    knowns_accuracy, OSE = knowns_accuracy[threshold_indices], OSE[threshold_indices]

    return knowns_accuracy, OSE
