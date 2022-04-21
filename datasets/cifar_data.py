import copy
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms

from paths import cifar_fpaths

from utils import load_pickle


def load_cifar_data(dataset, combine=True):
    """ Load Digits Data from pickle data
    params:
    @dataset: "cifar10", "cifar100"
    return:
    @xs: numpy.array, (n, c, w, h)
    @ys: numpy.array, (n, ), 0-9
    """
    train_xs, train_ys = [], []
    for fpath in cifar_fpaths[dataset]["train_fpaths"]:
        infos = load_pickle(fpath)
        train_xs.append(infos["images"])
        train_ys.append(infos["labels"])
    train_xs = np.concatenate(train_xs, axis=0)
    train_ys = np.concatenate(train_ys, axis=0)

    test_xs, test_ys = [], []
    infos = load_pickle(cifar_fpaths[dataset]["test_fpath"])
    test_xs = infos["images"]
    test_ys = infos["labels"]

    if combine:
        xs = np.concatenate([train_xs, test_xs], axis=0)
        ys = np.concatenate([train_ys, test_ys], axis=0)
        return xs, ys
    else:
        return train_xs, train_ys, test_xs, test_ys


class CifarDataset(data.Dataset):
    def __init__(self, xs, ys, is_train=True):
        self.xs = copy.deepcopy(xs)
        self.ys = copy.deepcopy(ys)

        if is_train is True:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        img = self.xs[index]
        label = self.ys[index]

        img = img.transpose((1, 2, 0)).astype(np.uint8)
        img = self.transform(img)

        img = torch.FloatTensor(img)
        label = torch.LongTensor([label])[0]
        return img, label
