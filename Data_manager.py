from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from ConcatDataset import ConcatDataset
from HiddenPrints import HiddenPrints


def mnist_vanilla(device):
    train = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    if device == "cpu":
        subtrain = list(range(1, 5001))
        subtest = list(range(1, 1001))
        training_data = Subset(train, subtrain)
        test_data = Subset(test, subtest)

        return training_data, test_data
    else:
        return train, test


def Concat_digit_letter(device, trainsamples = None, testsamples = None):
    letters_train = datasets.EMNIST(
        root="data",
        split="letters",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    letters_test = datasets.EMNIST(
        root="data",
        split="letters",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    digits_train = datasets.EMNIST(
        root="data",
        split="digits",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    digits_test = datasets.EMNIST(
        root="data",
        split="digits",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # no printing
    with HiddenPrints():
        training_data = ConcatDataset([digits_train, letters_train])
        test_data = ConcatDataset([digits_test, letters_test])

    # take different sizes of the datasets depending if GPU is available
    if device == "cpu":
        subtrain = list(range(1, 5001))
        subtest = list(range(1, 1001))
        training_data = Subset(training_data, subtrain)
        test_data = Subset(test_data, subtest)

        return training_data, test_data
    else:
        if trainsamples:
            subtrain = list(range(1, len(training_data) + 1, round((len(training_data) + 1) / trainsamples)))
            training_data = Subset(training_data, subtrain)
        if testsamples:
            subtest = list(range(1, len(test_data) + 1, round((len(test_data) + 1) / testsamples)))
            test_data = Subset(test_data, subtest)

        return training_data, test_data
