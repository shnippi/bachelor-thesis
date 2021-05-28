import os
from dotenv import load_dotenv

load_dotenv()


def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the main training script for all MNIST experiments. \
                    Where applicable roman letters are used as Known Unknowns. \
                    During training model with best performance on validation set in the no_of_epochs is used.'
    )

    parser.add_argument("--approach", default="entropic", required=False,
                        choices=['SoftMax', 'CenterLoss', 'COOL', 'BG', 'entropic', 'objectosphere'])
    parser.add_argument('--second_loss_weight', help='Loss weight for Objectosphere loss', type=float, default=0.0001)
    parser.add_argument('--minimum_prediction', type=float, default=0.9,
                        help="Select the minimum probability for a sample to generate an adversarial for it")
    parser.add_argument('--adversarial_strength', type=float, default=0.3,
                        help="Select the modification strength of the adversarial perturbation (smaller values get more difficult adversarials)")
    parser.add_argument('--Minimum_Knowns_Magnitude', help='Minimum Possible Magnitude for the Knowns', type=float,
                        default=50.)
    parser.add_argument("--solver", action="store", dest="solver", default='sgd', choices=['sgd', 'adam'])
    parser.add_argument("--lr", action="store", dest="lr", default=0.01, type=float)
    parser.add_argument('--batch_size', help='Batch_Size', action="store", dest="Batch_Size", type=int, default=128)
    parser.add_argument("--no_of_epochs", action="store", dest="no_of_epochs", type=int, default=70)
    parser.add_argument("--dataset_root", default="data", help="Select the directory where datasets are stored.")

    return parser.parse_args()


def main():
    # TODO: what does sphere do?
    print("begin")

    args = command_line_options()

    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.tensorboard import SummaryWriter
    from torch.nn import functional as F
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(0)

    # from vast import architectures
    # from vast import tools
    # from vast import losses

    from loss import entropic_openset_loss
    from LeNet_plus_plus import LeNet_plus_plus
    from ConcatDataset import ConcatDataset
    from metrics import accuracy_known, confidence, sphere
    from lots import lots_

    import random

    import pathlib
    import sys

    device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"

    mnist_trainset = torchvision.datasets.MNIST(
        root=args.dataset_root,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    letters_trainset = torchvision.datasets.EMNIST(
        root=args.dataset_root,
        train=True,
        download=True,
        split='letters',
        transform=transforms.ToTensor()
    )

    mnist_valset = torchvision.datasets.MNIST(
        root=args.dataset_root,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    letters_valset = torchvision.datasets.EMNIST(
        root=args.dataset_root,
        train=False,
        download=True,
        split='letters',
        transform=transforms.ToTensor()
    )

    def get_loss_functions(args):
        approach = {"SoftMax": dict(first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                                    dir_name="Softmax",
                                    training_data=[mnist_trainset],
                                    val_data=[mnist_valset]
                                    ),
                    # "CenterLoss": dict(first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    #                 second_loss_func=losses.tensor_center_loss(beta=0.1),
                    #                 dir_name = "CenterLoss",
                    #                 training_data = [mnist_trainset],
                    #                 val_data = [mnist_valset]
                    #                 ),
                    # "COOL": dict(first_loss_func=losses.entropic_openset_loss(),
                    #             second_loss_func=losses.objecto_center_loss(
                    #                 beta=0.1,
                    #                 classes=range(-1, 10, 1),
                    #                 ring_size=args.Minimum_Knowns_Magnitude),
                    #             dir_name = "COOL",
                    #             training_data = [mnist_trainset],
                    #             val_data = [mnist_valset, letters_valset]
                    #             ),
                    "BG": dict(first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                               second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                               dir_name="BGSoftmax",
                               training_data=[mnist_trainset],
                               val_data=[mnist_valset, letters_valset]
                               ),
                    "entropic": dict(first_loss_func=entropic_openset_loss(),
                                     second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                                     dir_name="Cross",
                                     training_data=[mnist_trainset],
                                     val_data=[mnist_valset, letters_valset]
                                     ),
                    # "objectosphere": dict(first_loss_func=losses.entropic_openset_loss(),
                    #                     second_loss_func=losses.objectoSphere_loss(
                    #                         args.Batch_Size,
                    #                         knownsMinimumMag=args.Minimum_Knowns_Magnitude),
                    #                     dir_name = "ObjectoSphere",
                    #                     training_data = [mnist_trainset],
                    #                     val_data = [mnist_valset, letters_valset]
                    #                     )
                    }
        return approach[args.approach]

    first_loss_func, second_loss_func, dir_name, training_data, validation_data = \
    list(zip(*get_loss_functions(args).items()))[-1]

    results_dir = pathlib.Path(dir_name)
    save_dir = results_dir / f"{dir_name}_{args.adversarial_strength:1.2f}_lots.model"
    results_dir.mkdir(parents=True, exist_ok=True)

    training_data = ConcatDataset(training_data, BG=args.approach == "BG")
    validation_data = ConcatDataset(validation_data, BG=args.approach == "BG")
    net = LeNet_plus_plus()
    net = net.to(device)
    train_data_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.Batch_Size,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.Batch_Size,
        pin_memory=True
    )

    if args.solver == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.solver == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    logs_dir = results_dir / 'Logs'
    writer = SummaryWriter(logs_dir)
    dataset_sizes = training_data.all_sizes

    for epoch in range(1, args.no_of_epochs + 1, 1):  # loop over the dataset multiple times
        loss_history = []
        train_accuracy = torch.zeros(2, dtype=int, device=device)
        train_magnitude = torch.zeros(2, dtype=float)
        train_confidence = torch.zeros(2, dtype=float, device=device)
        for b, (x, y) in enumerate(train_data_loader):

            #            if b % 10 == 0:
            #              optimizer.step()
            #              sys.stdout.write(".")
            #              sys.stdout.flush()

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, features = net(x, features=True)
            if epoch <= 3 and args.approach in ["COOL", "CenterLoss"]:
                loss = first_loss_func(logits, y)
            else:
                loss = first_loss_func(logits, y) + args.second_loss_weight * second_loss_func(features, y)

            if args.approach in ["CenterLoss", "COOL"]:
                second_loss_func.update_centers(features, y)

            # metrics on training set
            train_accuracy += accuracy_known(logits, y).to(device)
            train_confidence += confidence(logits, y)
            if args.approach not in ("SoftMax", "BG"):
                train_magnitude += sphere(features, y, args.Minimum_Knowns_Magnitude if args.approach in ("COOL", "Objectosphere") else None)

            loss_history.append(loss)
            loss.backward()

            ######################
            optimizer.step()
            optimizer.zero_grad()
            if args.approach != "SoftMax":
                # generate adversarial samples as negative class
                features = features.detach()

                # select samples that are correctly classified
                with torch.no_grad():
                    sm = torch.nn.functional.softmax(logits, dim=1)

                # targets = tools.device(torch.zeros((N, features.shape[1])))
                x_in = []
                t_in = []
                # batch version of LOTS
                for i, t in enumerate(y):
                    # check if this sample is sufficiently well classified
                    if sm[i, t] < args.minimum_prediction:
                        continue
                    # select a target with a different class from the batch
                    j = None
                    while j is None or y[j] == t:
                        j = random.randint(0, len(y) - 1)
                    x_in.append(x[j])
                    t_in.append(features[j])
                if not x_in:
                    # no sample classified sufficiently well
                    continue

                x_adv = lots_(net, torch.stack(x_in).to(device), torch.stack(t_in).to(device),
                              args.adversarial_strength)
                y_adv = torch.ones(len(x_adv), dtype=torch.long) * (10 if args.approach == "BG" else -1)
                y_adv = y_adv.to(device)

                #              import cv2
                #              for i, a,b,t in zip(range(len(y)), x, x_adv, y):
                #                cv2.imwrite("C:/temp/%d_%d_orig.png" % (i, int(t)), (a.detach().cpu().numpy()[0] * 255).astype("uint8"))
                #                cv2.imwrite("C:/temp/%d_%d_adv.png" % (i, int(t)), (b.detach().cpu().numpy()[0] * 255).astype("uint8"))
                #              import ipdb; ipdb.set_trace()

                # TODO: NO STEP HERE????

                # forward pass adversarial images
                optimizer.zero_grad()
                logits, features = net(x_adv, features=True)
                if epoch <= 3 and args.approach in ["COOL", "CenterLoss"]:
                    loss = first_loss_func(logits, y_adv)
                else:
                    loss = first_loss_func(logits, y_adv) + args.second_loss_weight * second_loss_func(features, y_adv)
                loss_history.append(loss)

                if args.approach in ["CenterLoss", "COOL"]:
                    second_loss_func.update_centers(features, y_adv)

                # metrics on training set
                train_accuracy += accuracy_known(logits, y_adv).to(device)
                train_confidence += confidence(logits, y_adv)
                # if args.approach not in ("SoftMax", "BG"):
                #     train_magnitude += losses.sphere(features, y_adv, args.Minimum_Knowns_Magnitude if args.approach in ("COOL", "Objectosphere") else None)

                # train on both original and adversarial images at the same time
                loss.backward()

        #       print()
        # metrics on validation set
        with torch.no_grad():
            val_loss = torch.zeros(2, dtype=float)
            val_accuracy = torch.zeros(2, dtype=int)
            val_magnitude = torch.zeros(2, dtype=float)
            val_confidence = torch.zeros(2, dtype=float, device=device)
            for x, y in val_data_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = net(x, features=True)

                loss = first_loss_func(outputs[0], y) + args.second_loss_weight * second_loss_func(outputs[1], y)
                val_loss += torch.tensor((torch.sum(loss), 1))
                val_accuracy += accuracy_known(outputs[0], y)
                val_confidence += confidence(outputs[0], y).to(device)
                if args.approach not in ("SoftMax", "BG"):
                    val_magnitude += sphere(outputs[1], y, args.Minimum_Knowns_Magnitude if args.approach in ("COOL", "Objectosphere") else None)

        epoch_running_loss = torch.mean(torch.tensor(loss_history))
        writer.add_scalar('Loss/train', epoch_running_loss, epoch)
        writer.add_scalar('Loss/val', val_loss[0] / val_loss[1], epoch)
        writer.add_scalar('Acc/train', float(train_accuracy[0]) / float(train_accuracy[1]), epoch)
        writer.add_scalar('Acc/val', float(val_accuracy[0]) / float(val_accuracy[1]), epoch)
        writer.add_scalar('Conf/train', float(train_confidence[0]) / float(train_confidence[1]), epoch)
        writer.add_scalar('Conf/val', float(val_confidence[0]) / float(val_confidence[1]), epoch)
        writer.add_scalar('Mag/train', train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else 0, epoch)
        writer.add_scalar('Mag/val', val_magnitude[0] / val_magnitude[1], epoch)

        save_status = "NO"
        if epoch <= 5:
            prev_confidence = None
        if prev_confidence is None or (val_confidence[0] > prev_confidence):
            torch.save(net.state_dict(), save_dir)
            prev_confidence = val_confidence[0]
            save_status = "YES"

        print(
            "Epoch %03d -- train-loss: %1.5f  accuracy: %1.5f  confidence: %1.5f  magnitude: %1.5f  --  val-loss: %1.5f  accuracy: %1.5f  confidence: %1.5f  magnitude: %1.5f -- saving model: %s" % (
                epoch,
                epoch_running_loss,
                float(train_accuracy[0]) / float(train_accuracy[1]),
                train_confidence[0] / train_confidence[1],
                train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else 0,
                float(val_loss[0]) / float(val_loss[1]),
                float(val_accuracy[0]) / float(val_accuracy[1]),
                val_confidence[0] / val_confidence[1],
                val_magnitude[0] / val_magnitude[1] if val_magnitude[1] else 0,
                save_status
            ))


if __name__ == "__main__":
    main()
