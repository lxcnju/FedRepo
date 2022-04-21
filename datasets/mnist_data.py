import os
import copy
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms

from paths import digits_fdir

from utils import load_pickle


def load_mnist_data(dataset, combine=True):
    train_fpath = os.path.join(
        digits_fdir, "mnist-train-raw.pkl"
    )

    test_fpath = os.path.join(
        digits_fdir, "mnist-test-raw.pkl"
    )

    train_obj = load_pickle(train_fpath)
    test_obj = load_pickle(test_fpath)
    train_xs = train_obj["images"]
    train_ys = train_obj["labels"]
    test_xs = test_obj["images"]
    test_ys = test_obj["labels"]

    if combine:
        xs = np.concatenate([train_xs, test_xs], axis=0)
        ys = np.concatenate([train_ys, test_ys], axis=0)
        return xs, ys
    else:
        return train_xs, train_ys, test_xs, test_ys


class MnistDataset(data.Dataset):
    def __init__(self, xs, ys, is_train=None):
        self.xs = copy.deepcopy(xs)
        self.ys = copy.deepcopy(ys)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            )
        ])

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        raw_img = self.xs[index]
        label = self.ys[index]

        # transforms.ToPILImage need (H, W, C) np.uint8 input
        img = raw_img.transpose(1, 2, 0).astype(np.uint8)

        # return (C, H, W) tensor
        img = self.transform(img)

        label = torch.LongTensor([label])[0]
        return img, label
