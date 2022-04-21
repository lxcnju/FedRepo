import os
import pandas as pd
import pickle
import json
import random
import numpy as np

import torch
import torch.nn as nn

try:
    import moxing as mox
    open = mox.file.File
except Exception:
    pass


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_file(path):
    da_df = pd.read_csv(
        path, index_col=False, header=None
    )
    return da_df


def save_data(da_df, path):
    da_df.to_csv(path)
    print("File saved in {}.".format(path))


def load_pickle(fpath):
    with open(fpath, "rb") as fr:
        data = pickle.load(fr)
    return data


def save_pickle(data, fpath):
    with open(fpath, "wb") as fw:
        pickle.dump(data, fw)
    return data


def load_json(fpath):
    with open(fpath, "r") as fr:
        data = json.load(fr)
    return data


def save_json(data, fpath):
    with open(fpath, "w") as fr:
        data = json.dump(data, fr)
    return data


def append_to_logs(fpath, logs):
    with open(fpath, "a", encoding="utf-8") as fa:
        for log in logs:
            fa.write("{}\n".format(log))
        fa.write("\n")


def format_logs(logs):
    def formal_str(x):
        if isinstance(x, int):
            return str(x)
        elif isinstance(x, float):
            return "{:.5f}".format(x)
        else:
            return str(x)

    logs_str = []
    for key, elems in logs.items():
        log_str = "[{}]: ".format(key)
        log_str += " ".join([formal_str(e) for e in elems])
        logs_str.append(log_str)
    return logs_str


def listfiles(fdir):
    for root, dirs, files in os.walk(fdir):
        print(root, dirs, files)


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def prediction_mask(logits, label):
    pred = torch.argmax(logits, dim=1)
    mask = (pred == label).float()
    return mask


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
            m.weight, mode='fan_out', nonlinearity='relu'
        )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        try:
            nn.init.constant_(m.bias, 0)
        except Exception:
            pass
