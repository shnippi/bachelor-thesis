from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from ConcatDataset import ConcatDataset
from helper import HiddenPrints


def make_subset(device, training_data, test_data, trainsamples=None, testsamples=None):
    # take different sizes of the datasets depending if GPU is available

    if trainsamples:
        subtrain = list(range(1, len(training_data) + 1, round((len(training_data) + 1) / trainsamples)))
        training_data = Subset(training_data, subtrain)
    if testsamples:
        subtest = list(range(1, len(test_data) + 1, round((len(test_data) + 1) / testsamples)))
        test_data = Subset(test_data, subtest)

    return training_data, test_data


# vanilla MNIST
def mnist(device, trainsamples=None, testsamples=None):
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

    return make_subset(device, train, test, trainsamples, testsamples)


# digits from EMNIST
def emnist_digits(device, trainsamples=None, testsamples=None):
    train = datasets.EMNIST(
        root="data",
        split="digits",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test = datasets.EMNIST(
        root="data",
        split="digits",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return make_subset(device, train, test, trainsamples, testsamples)


# digits from MNIST concatenated with EMNIST letters
def mnist_plus_letter(device, trainsamples=None, testsamples=None):
    # training set
    digits_train = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    letters_train = datasets.EMNIST(
        root="data",
        split="letters",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # testing set
    digits_test = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    letters_test = datasets.EMNIST(
        root="data",
        split="letters",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # no printing
    with HiddenPrints():
        training_data = ConcatDataset([digits_train, letters_train])
        test_data = ConcatDataset([digits_test, letters_test])

    return make_subset(device, training_data, test_data, trainsamples, testsamples)


# digits and letters from EMNIST
def concat_emnist(device, trainsamples=None, testsamples=None):
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

    return make_subset(device, training_data, test_data, trainsamples, testsamples)


# openset dataset with MNIST digits for training and MNIST digits plus EMNIST letters for testing
def open_set(device, trainsamples=None, testsamples=None):
    # training set
    letters_train = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # testing set
    digits_test = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    letters_test = datasets.EMNIST(
        root="data",
        split="letters",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # no printing
    with HiddenPrints():
        test_data = ConcatDataset([digits_test, letters_test])
        training_data = ConcatDataset([letters_train])

    return make_subset(device, training_data, test_data, trainsamples, testsamples)
