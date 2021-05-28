def command_line_options():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='TODO: Write this'
    )

    parser.add_argument("--arch", default='LeNet_plus_plus', required=False,
                        choices=['LeNet', 'LeNet_plus_plus'])
    # TODO: make path dynamic
    parser.add_argument("--model_file_name", action="store", default='Cross/Cross_0.30_lots.model')
    parser.add_argument("--save_features", action="store_true", default=False)
    parser.add_argument("--BG_class", action="store_true", default=False)
    parser.add_argument("--Sigmoid_Plotter", action="store_true", default=False)
    parser.add_argument("--run_on_cpu", action="store_true", help = "If selected, features are extracted in the CPU")
    parser.add_argument("--dataset_root", default ="data", help="Select the directory where datasets are stored.")

    return parser.parse_args()


import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms


# from vast.tools import viz
# from vast import architectures, tools


import pandas as pd
from LeNet_plus_plus import LeNet_plus_plus
import visualization as viz

import pathlib

import csv
from dotenv import load_dotenv
load_dotenv()
import  os



device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"


def extract_features(args, model_file_name, data_obj, use_BG=False):
    net = LeNet_plus_plus()
    net.load_state_dict(torch.load(model_file_name,map_location='cuda:0'))
    net = net.to(device)

    # TODO: call eval?

    data_loader = torch.utils.data.DataLoader(data_obj, batch_size=2048, shuffle=False,
                                             pin_memory=True)

    gt = []
    fetures = []
    logits = []

    for (x, y) in data_loader:
        gt.extend(y)
        x = x.to(device)
        output = net(x, features = True)
        fetures.extend(output[1].tolist())
        logits.extend(output[0].tolist())
    del net
    return torch.tensor(gt), torch.tensor(fetures), torch.tensor(logits)


def main():

    args = command_line_options()

    # if args.run_on_cpu:
    #     tools.set_device_cpu()

    mnist_testset = torchvision.datasets.MNIST(
        root=args.dataset_root,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    letters_testset = torchvision.datasets.EMNIST(
        root=args.dataset_root,
        train=False,
        download=True,
        split='letters',
        transform=transforms.ToTensor()
    )
    fashion_testset = torchvision.datasets.FashionMNIST(root=args.dataset_root, train=False,
                                                        transform=transforms.ToTensor(), download=True)

    if args.Sigmoid_Plotter:
        plotter = viz.sigmoid_2D_plotter
    else:
        plotter = viz.plotter_2D
    if args.arch!='LeNet_plus_plus':
        plotter = lambda arg1, arg2, *args, **kwargs: None

    model_file_name = pathlib.Path(args.model_file_name)
    pos_gt,pos_feat,pos_logits = extract_features(args, model_file_name,mnist_testset,args.BG_class)
    plotter(pos_feat.numpy(),
            pos_gt.numpy(),
            title=None,
            file_name=str(model_file_name.parent/'digits_{}.{}'),
            final=False,
            pred_weights=None,
            heat_map=False
            )


    if args.save_features:
        df = pd.DataFrame({'GT': pos_gt.numpy(),
                           'Features1': pos_feat.numpy()[:,0],
                           'Features2': pos_feat.numpy()[:,1]})
        df.to_csv(model_file_name.parent/'mnist.csv',index=False)

    pos_softmax = F.softmax(pos_logits,dim=1)
    scores_data = torch.cat((pos_gt.type(torch.FloatTensor)[:,None],pos_softmax),dim=1)
    header = ['GT']
    header.extend([f'Class_{_}' for _ in range(pos_softmax.shape[1])])
    scores_data = [header] + scores_data.tolist()
    with open(model_file_name.parent/'mnist_scores.csv','w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows(scores_data)


    letters_gt,letters_feat,letters_logits = extract_features(args, model_file_name,letters_testset,args.BG_class)
    plotter(pos_feat.numpy(),
            pos_gt.numpy(),
            neg_features = letters_feat.numpy(),
            neg_labels = letters_gt.numpy(),
            title=None,
            file_name=str(model_file_name.parent/'letters_{}.{}'),
            final=False,
            pred_weights=None,
            heat_map=False
            )



    if args.save_features:
        df = pd.DataFrame({'GT': letters_gt.numpy(),
                           'Features1': letters_feat.numpy()[:,0],
                           'Features2': letters_feat.numpy()[:,1]})
        df.to_csv(model_file_name.parent/'letters.csv',index=False)

    letters_softmax = F.softmax(letters_logits,dim=1)
    scores_data = torch.cat((letters_gt.type(torch.FloatTensor)[:,None],letters_softmax),dim=1)
    header = ['GT']
    header.extend([f'Class_{_}' for _ in range(letters_softmax.shape[1])])
    scores_data = [header] + scores_data.tolist()
    with open(model_file_name.parent/'letters_scores.csv','w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows(scores_data)




    fashion_gt,fashion_feat,fashion_logits = extract_features(args, model_file_name,fashion_testset,args.BG_class)
    plotter(pos_feat.numpy(),
            pos_gt.numpy(),
            neg_features = fashion_feat.numpy(),
            neg_labels = fashion_gt.numpy(),
            title=None,
            file_name=str(model_file_name.parent/'fashion_{}.{}'),
            final=False,
            pred_weights=None,
            heat_map=False
            )



    if args.save_features:
        df = pd.DataFrame({'GT': fashion_gt.numpy(),
                           'Features1': fashion_feat.numpy()[:,0],
                           'Features2': fashion_feat.numpy()[:,1]})
        df.to_csv(model_file_name.parent/'fashion.csv',index=False)

    fashion_softmax = F.softmax(fashion_logits,dim=1)
    scores_data = torch.cat((fashion_gt.type(torch.FloatTensor)[:,None],fashion_softmax),dim=1)
    header = ['GT']
    header.extend([f'Class_{_}' for _ in range(fashion_softmax.shape[1])])
    scores_data = [header] + scores_data.tolist()
    with open(model_file_name.parent/'fashion_scores.csv','w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows(scores_data)


if __name__ == "__main__":
    main()