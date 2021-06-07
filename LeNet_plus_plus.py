import torch.nn as nn

# TODO: change batchnorm (momentum, etc)
# TODO: track running stats = Flase if no solution

class LeNet_plus_plus(nn.Module):
    def __init__(self):
        super(LeNet_plus_plus, self).__init__()
        # first convolution block
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(in_channels=self.conv1_1.out_channels, out_channels=32, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm1 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # second convolution block
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_2.out_channels, out_channels=64, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv2_1.out_channels, out_channels=64, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm2 = nn.BatchNorm2d(self.conv2_2.out_channels)
        # third convolution block
        self.conv3_1 = nn.Conv2d(in_channels=self.conv2_2.out_channels, out_channels=128, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.conv3_2 = nn.Conv2d(in_channels=self.conv3_1.out_channels, out_channels=128, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm3 = nn.BatchNorm2d(self.conv3_2.out_channels)
        # fully-connected layers
        self.fc1 = nn.Linear(in_features=self.conv3_2.out_channels * 3 * 3,
                             out_features=2, bias=True)
        self.fc2 = nn.Linear(in_features=2, out_features=10, bias=True)
        # activation function
        self.prelu_act = nn.PReLU()

    def forward(self, x, features=False):
        # compute first convolution block output
        x = self.prelu_act(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        # compute second convolution block output
        x = self.prelu_act(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        # compute third convolution block output
        x = self.prelu_act(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        # turn into 1D representation (1D per batch element)
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)
        # first fully-connected layer to compute 2D feature space
        z = self.fc1(x)
        # second fully-connected layer to compute the logits
        y = self.fc2(z)
        if features:
            # return both the logits and the deep features
            return y, z
        else:
            return y
