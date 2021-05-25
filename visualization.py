import numpy as np
import itertools

import torch
from matplotlib import pyplot as plt
import os
from sklearn.metrics import roc_auc_score

colors = np.array([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [67, 99, 216],
    [245, 130, 49],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [188, 246, 12],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [154, 99, 36],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
    [128, 128, 0],
    [255, 216, 177],
    [0, 0, 117],
    [128, 128, 128],
    [255, 255, 255],
    [0, 0, 0]
]).astype(np.float)
colors = colors / 255.


def roc(pred, y):
    scores = torch.ones_like(y, dtype=torch.float)
    target = torch.ones_like(y)
    for i in range(len(y)):
        if y[i] == -1:
            scores[i] = torch.max(pred[i])
            target[i] = 0
        else:
            scores[i] = pred[i][y[i]]

    scores = scores.detach().numpy()
    target = target.detach().numpy()

    # TODO: ovr vs ovo??
    return roc_auc_score(target, scores, multi_class="ovo")


def simplescatter(features, classes, eps=None, eps_iter=None, current_iteration=None,
                  c=("b", "g", "r", "c", "m", "y", "orange", "lawngreen", "peru", "deeppink", "k"),
                  s=0.1):
    plt.figure(1)
    # scatterplot every digit to a color
    for i in range(classes):
        plt.scatter(*zip(*(features[i])), c=c[i], s=s)

    plt.legend(range(classes))

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    if eps and eps_iter and current_iteration:
        plt.savefig(f"plots/flower_{eps}eps_{eps_iter}epsiter_{current_iteration}iter.png", dpi=600)
    else:
        plt.savefig("plots/flower.png", dpi=600)

    if os.environ.get('PLOT') == "t":
        plt.show()
    plt.close()


def epsilon_plot(eps_tensor, eps_list, eps_iter_list, iteration=None):
    plt.figure(2)
    # pull out the 3rd (depth) dimension of the tensor. Now for every eps-eps_iter pair theres a list with
    # confidences over all epochs
    confidences = eps_tensor.reshape(len(eps_tensor), -1).transpose(0, 1)
    max_conf = 0

    # TODO: fix the legend

    for i in range(len(confidences)):
        eps_index = i // len(eps_iter_list)
        eps_iter_index = i % len(eps_iter_list)
        if confidences[i][-1] > max_conf and confidences[i][-1] > 0.5:
            plt.plot(confidences[i], label=f"eps: {eps_list[eps_index]}, eps_iter: {eps_iter_list[eps_iter_index]}")
            max_conf = confidences[i][-1]
        elif confidences[i][-1] > 0.5:
            plt.plot(confidences[i])

    plt.xlabel("epochs")
    plt.ylabel("confidence")
    plt.legend()

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    if iteration:
        plt.savefig(f"plots/epsilons_iter{iteration}.png")
    else:
        plt.savefig("plots/epsilons.png")

    if os.environ.get('PLOT') == "t":
        plt.show()
    plt.close()


# TODO: make like the max the most saturated and then fading stuff
def epsilon_table(eps_tensor, eps_list, eps_iter_list, iteration=None):
    plt.figure(figsize=(6, 4), dpi=800)

    data = eps_tensor[-1].cpu().detach().numpy()
    max_idx = np.unravel_index(data.argmax(), data.shape)

    cell_colors = np.full(data.shape, "w")
    cell_colors[max_idx[0]][max_idx[1]] = "g"

    columns = eps_iter_list
    rows = eps_list

    plt.axis('tight')
    plt.axis('off')

    plt.title("confidences")

    plt.table(cellText=data, rowLabels=rows, colLabels=columns, loc="center", cellColours=cell_colors)

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    if iteration:
        plt.savefig(f"plots/table_iter{iteration}.png")
    else:
        plt.savefig("plots/table.png")

    if os.environ.get('PLOT') == "t":
        plt.show()
    plt.close()


def plot_histogram(pos_features, neg_features, pos_labels='Knowns', neg_labels='Unknowns', title="Histogram",
                   file_name='{}foo.pdf'):
    """
    This function plots the Histogram for Magnitudes of feature vectors.
    """
    pos_mag = np.sqrt(np.sum(np.square(pos_features), axis=1))
    neg_mag = np.sqrt(np.sum(np.square(neg_features), axis=1))

    pos_hist = np.histogram(pos_mag, bins=500)
    neg_hist = np.histogram(neg_mag, bins=500)

    fig, ax = plt.subplots(figsize=(4.5, 1.75))
    ax.plot(pos_hist[1][1:], pos_hist[0].astype(np.float16) / max(pos_hist[0]), label=pos_labels, color='g')
    ax.plot(neg_hist[1][1:], neg_hist[0].astype(np.float16) / max(neg_hist[0]), color='r', label=neg_labels)

    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.xscale('log')
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.savefig(file_name.format('Hist', 'pdf'), bbox_inches='tight')

    plt.show()
    plt.close()


def plotter_2D(
        pos_features,
        labels,
        neg_features=None,
        pos_labels='Knowns',
        neg_labels='Unknowns',
        title=None,
        file_name='foo.pdf',
        final=False,
        pred_weights=None,
        heat_map=False):
    global colors
    plt.figure(figsize=[6, 6])

    if heat_map:
        min_x, max_x = np.min(pos_features[:, 0]), np.max(pos_features[:, 0])
        min_y, max_y = np.min(pos_features[:, 1]), np.max(pos_features[:, 1])
        x = np.linspace(min_x * 1.5, max_x * 1.5, 200)
        y = np.linspace(min_y * 1.5, max_y * 1.5, 200)
        pnts = list(itertools.chain(itertools.product(x, y)))
        pnts = np.array(pnts)

        e_ = np.exp(np.dot(pnts, pred_weights))
        e_ = e_ / np.sum(e_, axis=1)[:, None]
        res = np.max(e_, axis=1)

        plt.pcolor(x, y, np.array(res).reshape(200, 200).transpose(), rasterized=True)

    if neg_features is not None:
        # Remove black color from knowns
        colors = colors[:-1, :]

    # TODO:The following code segment needs to be improved
    colors_with_repetition = colors.tolist()
    for i in range(int(len(set(labels.tolist())) / colors.shape[0])):
        colors_with_repetition.extend(colors.tolist())
    colors_with_repetition.extend(colors.tolist()[:int(colors.shape[0] % len(set(labels.tolist())))])
    colors_with_repetition = np.array(colors_with_repetition)

    plt.scatter(pos_features[:, 0], pos_features[:, 1], c=colors_with_repetition[labels.astype(np.int)],
                edgecolors='none', s=0.5)
    if neg_features is not None:
        plt.scatter(neg_features[:, 0], neg_features[:, 1], c='k', edgecolors='none', s=15, marker="*")
    if final:
        plt.gca().spines['right'].set_position('zero')
        plt.gca().spines['bottom'].set_position('zero')
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labeltop=False, labelleft=False,
                        labelright=False)
        plt.axis('equal')

    plt.savefig(file_name.format('2D_plot', 'png'), bbox_inches='tight')
    plt.show()
    if neg_features is not None:
        plot_histogram(pos_features, neg_features, pos_labels=pos_labels, neg_labels=neg_labels, title=title,
                       file_name=file_name.format('hist', 'pdf'))


def sigmoid_2D_plotter(
        pos_features,
        labels,
        neg_features=None,
        pos_labels='Knowns',
        neg_labels='Unknowns',
        title=None,
        file_name='foo.pdf',
        final=False,
        pred_weights=None,
        heat_map=False):
    global colors
    plt.figure(figsize=[6, 6])

    if heat_map:
        min_x, max_x = np.min(pos_features[:, 0]), np.max(pos_features[:, 0])
        min_y, max_y = np.min(pos_features[:, 1]), np.max(pos_features[:, 1])
        x = np.linspace(min_x * 1.5, max_x * 1.5, 200)
        y = np.linspace(min_y * 1.5, max_y * 1.5, 200)
        pnts = list(itertools.chain(itertools.product(x, y)))
        pnts = np.array(pnts)

        e_ = np.exp(np.dot(pnts, pred_weights))
        e_ = e_ / np.sum(e_, axis=1)[:, None]
        res = np.max(e_, axis=1)

        plt.pcolor(x, y, np.array(res).reshape(200, 200).transpose(), rasterized=True)

    if neg_features is not None:
        # Remove black color from knowns
        colors = colors[:-1, :]

    colors_with_repetition = colors.tolist()
    for i in range(10):
        plt.scatter(pos_features[labels == i, 0], pos_features[labels == i, 1], c=colors_with_repetition[i],
                    edgecolors='none', s=1. - (i / 10))
    if neg_features is not None:
        plt.scatter(neg_features[:, 0], neg_features[:, 1], c='k', edgecolors='none', s=15, marker="*")
    if final:
        plt.gca().spines['right'].set_position('zero')
        plt.gca().spines['bottom'].set_position('zero')
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labeltop=False, labelleft=False,
                        labelright=False)
        plt.axis('equal')

    plt.savefig(file_name.format('2D_plot', 'png'), bbox_inches='tight')
    plt.show()
    if neg_features is not None:
        plot_histogram(pos_features, neg_features, pos_labels=pos_labels, neg_labels=neg_labels, title=title,
                       file_name=file_name.format('hist', 'pdf'))


def plot_OSRC(to_plot, no_of_false_positives=None, filename=None, title=None):
    """
    :param to_plot: list of tuples containing (knowns_accuracy,OSE,label_name)
    :param no_of_false_positives: To write on the x axis
    :param filename: filename to write
    :return: None
    """
    fig, ax = plt.subplots()
    if title != None:
        fig.suptitle(title, fontsize=20)
    for plot_no, (knowns_accuracy, OSE, label_name) in enumerate(to_plot):
        ax.plot(OSE, knowns_accuracy, label=label_name)
    ax.set_xscale('log')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Correct Classification Rate', fontsize=18, labelpad=10)
    if no_of_false_positives is not None:
        ax.set_xlabel(f"False Positive Rate : Total Unknowns {no_of_false_positives}", fontsize=18, labelpad=10)
    else:
        ax.set_xlabel(f"False Positive Rate", fontsize=18, labelpad=10)
    ax.legend(loc='lower center', bbox_to_anchor=(-1.25, 0.), ncol=1, fontsize=18, frameon=False)
    # ax.legend(loc="upper left")
    if filename is not None:
        fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.show()
