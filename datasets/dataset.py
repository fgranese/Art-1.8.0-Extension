import os
import sys
import inspect
import numpy as np
from art.utils import load_cifar10
from utils.utils_general import from_numpy_to_dataloader, from_dataloader_to_numpy

CIFAR10 = 'CIFAR10'
SVHN = 'SVHN'
CIFAR100 = 'CIFAR100'


def prep_folder(path: str, to_file: bool = False):
    if to_file:
        path_to_list_of_strings = path.split("/")
        if len(path_to_list_of_strings) < 2:
            sys.exit("Illegal path to file in function " + inspect.stack()[0][3])
        tmp = ""
        for i in range(len(path_to_list_of_strings) - 1):
            tmp += path_to_list_of_strings[i] + "/"
        path = tmp
    os.makedirs(path, exist_ok=True)


def load_svhn_data():
    from subprocess import call
    import scipy.io as sio

    if not os.path.isfile("data/svhn/train_32x32.mat"):
        os.makedirs("data/svhn/", exist_ok=True)
        print('Downloading SVHN train set...')
        call(
            "curl -o data/svhn/train_32x32.mat "
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            shell=True
        )
    if not os.path.isfile("data/svhn/test_32x32.mat"):
        os.makedirs("data/svhn/", exist_ok=True)
        print('Downloading SVHN test set...')
        call(
            "curl -o data/svhn/test_32x32.mat "
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            shell=True
        )
    train = sio.loadmat('data/svhn/train_32x32.mat')
    test = sio.loadmat('data/svhn/test_32x32.mat')
    x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
    y_train = np.reshape(train['y'], (-1,))
    y_test = np.reshape(test['y'], (-1,))
    np.place(y_train, y_train == 10, 0)
    np.place(y_test, y_test == 10, 0)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.reshape(x_train, (73257, 32, 32, 3))
    x_test = np.reshape(x_test, (26032, 32, 32, 3))

    min = x_test.min()
    max = x_test.max()

    return (x_train, y_train), (x_test, y_test), min, max


def get_dataloader_from_dataset_name(dataset_name: str, batch_size: int, train: bool, shuffle=False, return_numpy=False):
    dataset_name = dataset_name.upper()
    if dataset_name == CIFAR10:
        dataloader = get_CIFAR10(batch_size=batch_size, train=train, shuffle=shuffle, return_numpy=return_numpy)
    elif dataset_name == SVHN:
        dataloader = get_SVHN(batch_size=batch_size, train=train, shuffle=shuffle, return_numpy=return_numpy)
    elif dataset_name == CIFAR100:
        dataloader = get_CIFAR100(batch_size=batch_size, train=train, shuffle=shuffle, return_numpy=return_numpy)
    else:
        sys.exit('Requested dataset not available.')
    return dataloader


def get_CIFAR10(batch_size: int, train=False, shuffle=False, return_numpy=False):
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)

    if train:
        if return_numpy:
            return [x_train, y_train]
        else:
            return from_numpy_to_dataloader(X=x_train, y=y_train, batch_size=batch_size, shuffle=shuffle)
    else:
        if return_numpy:
            return [x_test, y_test]
        else:
            return from_numpy_to_dataloader(X=x_test, y=y_test, batch_size=batch_size, shuffle=shuffle)


def get_SVHN(batch_size: int, train=False, shuffle=False, return_numpy=False):
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_svhn_data()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32) / 255.
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32) / 255.

    if train:
        if return_numpy:
            return [x_train, y_train]
        else:
            return from_numpy_to_dataloader(X=x_train, y=y_train, batch_size=batch_size, shuffle=shuffle)
    else:
        if return_numpy:
            return [x_test, y_test]
        else:
            return from_numpy_to_dataloader(X=x_test, y=y_test, batch_size=batch_size, shuffle=shuffle)

def get_CIFAR100(batch_size: int, train=False, shuffle=False, return_numpy=False):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    data_man = datasets.CIFAR100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if train:
        train_dataset = data_man('data/', train=True, download=True, transform=transform_train)
        train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        if return_numpy:
            x_train = from_dataloader_to_numpy(train_set)
            x_train = x_train.reshape((x_train.shape[0] * x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4]))
            y_train = np.asarray(train_dataset.targets)
            return [x_train, y_train]
        else:
            return train_set
    else:
        test_dataset = data_man('data/', train=False, download=True, transform=transform_test)
        test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        if return_numpy:
            x_test = from_dataloader_to_numpy(test_set)
            x_test = x_test.reshape((x_test.shape[0] * x_test.shape[1], x_test.shape[2], x_test.shape[3], x_test.shape[4]))
            y_test = np.asarray(test_dataset.targets)
            return [x_test, y_test]
        else:
            return test_set


def get_dataloader(data_name: str, train: bool, batch_size=1, shuffle=False, *args, **kwargs):
    """Returns a dataloader given a dataset"""
    if 'return_numpy' in kwargs.keys():
        return_numpy = kwargs['return_numpy']
    else:
        return_numpy = False
    return get_dataloader_from_dataset_name(dataset_name=data_name, batch_size=batch_size, shuffle=shuffle, train=train, return_numpy=return_numpy)
