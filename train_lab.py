import os
import random
from collections import namedtuple
import numpy as np

import torch

from datasets.feddata import FedData

from algorithms.fedavg import FedAvg
from algorithms.fedreg import FedReg
from algorithms.scaffold import Scaffold

from algorithms.fedopt import FedOpt
from algorithms.fednova import FedNova
from algorithms.fedaws import FedAws
from algorithms.moon import MOON
from algorithms.feddyn import FedDyn

from algorithms.pfedme import pFedMe
from algorithms.perfedavg import PerFedAvg

from algorithms.fedrs import FedRS
from algorithms.fedphp import FedPHP
from algorithms.scaffoldrs import ScaffoldRS

from networks.basic_nets import get_basic_net
from networks.basic_nets import ClassifyNet

from paths import save_dir
from config import default_param_dicts
from utils import setup_seed

torch.set_default_tensor_type(torch.FloatTensor)


def construct_model(args):
    try:
        input_size = args.input_size
    except Exception:
        input_size = None

    try:
        input_channel = args.input_channel
    except Exception:
        input_channel = None

    model = get_basic_net(
        net=args.net,
        n_classes=args.n_classes,
        input_size=input_size,
        input_channel=input_channel,
    )

    model = ClassifyNet(
        net=args.net,
        init_way="none",
        n_classes=args.n_classes
    )

    return model


def construct_algo(args):
    if args.algo == "fedavg":
        FedAlgo = FedAvg
    elif args.algo == "fedprox":
        FedAlgo = FedReg
    elif args.algo == "fedmmd":
        FedAlgo = FedReg
    elif args.algo == "scaffold":
        FedAlgo = Scaffold
    elif args.algo == "fedopt":
        FedAlgo = FedOpt
    elif args.algo == "fednova":
        FedAlgo = FedNova
    elif args.algo == "fedaws":
        FedAlgo = FedAws
    elif args.algo == "moon":
        FedAlgo = MOON
    elif args.algo == "feddyn":
        FedAlgo = FedDyn
    elif args.algo == "pfedme":
        FedAlgo = pFedMe
    elif args.algo == "perfedavg":
        FedAlgo = PerFedAvg
    elif args.algo == "fedrs":
        FedAlgo = FedRS
    elif args.algo == "fedphp":
        FedAlgo = FedPHP
    elif args.algo == "scaffoldrs":
        FedAlgo = ScaffoldRS
    else:
        raise ValueError("No such fed algo:{}".format(args.algo))
    return FedAlgo


def get_hypers(algo):
    if algo == "fedavg":
        hypers = {
            "cnt": 2,
            "none": ["none"] * 2
        }
    elif algo == "fedprox":
        hypers = {
            "cnt": 2,
            "reg_way": ["fedprox"] * 2,
            "reg_lamb": [1e-5, 1e-1]
        }
    elif algo == "fedmmd":
        hypers = {
            "cnt": 2,
            "reg_way": ["fedmmd"] * 2,
            "reg_lamb": [1e-2, 1e-3]
        }
    elif algo == "scaffold":
        hypers = {
            "cnt": 2,
            "glo_lr": [0.25, 0.5]
        }
    elif algo == "fedopt":
        hypers = {
            "cnt": 2,
            "glo_optimizer": ["SGD", "Adam"],
            "glo_lr": [0.1, 3e-4],
        }
    elif algo == "fednova":
        hypers = {
            "cnt": 2,
            "gmf": [0.5, 0.1],
            "prox_mu": [1e-3, 1e-3],
        }
    elif algo == "fedaws":
        hypers = {
            "cnt": 2,
            "margin": [0.8, 0.5],
            "aws_steps": [30, 50],
            "aws_lr": [0.1, 0.01],
        }
    elif algo == "moon":
        hypers = {
            "cnt": 2,
            "reg_lamb": [1e-4, 1e-2]
        }
    elif algo == "feddyn":
        hypers = {
            "cnt": 2,
            "reg_lamb": [1e-3, 1e-2]
        }
    elif algo == "pfedme":
        hypers = {
            "cnt": 2,
            "reg_lamb": [1e-4, 1e-2],
            "alpha": [0.1, 0.75],
            "k_step": [20, 10],
            "beta": [1.0, 1.0],
        }
    elif algo == "perfedavg":
        hypers = {
            "cnt": 2,
            "meta_lr": [0.05, 0.01],
        }
    elif algo == "fedrs":
        hypers = {
            "cnt": 3,
            "alpha": [0.9, 0.5, 0.1],
        }
    elif algo == "fedphp":
        hypers = {
            "cnt": 3,
            "reg_way": ["KD", "MMD", "MMD"],
            "reg_lamb": [0.05, 0.1, 0.05],
        }
    elif algo == "scaffoldrs":
        hypers = {
            "cnt": 3,
            "glo_lr": [0.5, 0.25, 0.1],
            "alpha": [0.25, 0.1, 0.5],
        }
    else:
        raise ValueError("No such fed algo:{}".format(algo))
    return hypers


def main_federated(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # DataSets
    try:
        n_clients = args.n_clients
    except Exception:
        n_clients = None

    try:
        nc_per_client = args.nc_per_client
    except Exception:
        nc_per_client = None

    try:
        dir_alpha = args.dir_alpha
    except Exception:
        dir_alpha = None

    feddata = FedData(
        dataset=args.dataset,
        split=args.split,
        n_clients=n_clients,
        nc_per_client=nc_per_client,
        dir_alpha=dir_alpha,
        n_max_sam=args.n_max_sam,
    )
    csets, gset = feddata.construct()

    try:
        nc = int(args.dset_ratio * len(csets))
        clients = list(csets.keys())
        sam_clients = np.random.choice(
            clients, nc, replace=False
        )
        csets = {
            c: info for c, info in csets.items() if c in sam_clients
        }

        n_test = int(args.dset_ratio * len(gset.xs))
        inds = np.random.permutation(len(gset.xs))
        gset.xs = gset.xs[inds[0:n_test]]
        gset.ys = gset.ys[inds[0:n_test]]

    except Exception:
        pass

    feddata.print_info(csets, gset)

    # Model
    model = construct_model(args)
    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        model = model.cuda()

    FedAlgo = construct_algo(args)
    algo = FedAlgo(
        csets=csets,
        gset=gset,
        model=model,
        args=args
    )
    algo.train()

    fpath = os.path.join(
        save_dir, args.fname
    )
    algo.save_logs(fpath)
    print(algo.logs)


def main_cifar_label(dataset, algo):
    hypers = get_hypers(algo)

    lr = 0.03
    for net in ["TFCNN", "VGG11", "VGG11-BN"]:
        for local_epochs in [2, 5]:
            for j in range(hypers["cnt"]):
                para_dict = {}
                for k, vs in default_param_dicts[dataset].items():
                    para_dict[k] = random.choice(vs)

                para_dict["algo"] = algo
                para_dict["dataset"] = dataset
                para_dict["net"] = net
                para_dict["split"] = "label"

                if dataset == "cifar10":
                    para_dict["nc_per_client"] = 2
                elif dataset == "cifar100":
                    para_dict["nc_per_client"] = 20

                para_dict["lr"] = lr
                para_dict["n_clients"] = 100
                para_dict["c_ratio"] = 0.1
                para_dict["local_epochs"] = local_epochs
                para_dict["max_round"] = 1000
                para_dict["test_round"] = 10

                for key, values in hypers.items():
                    if key == "cnt":
                        continue
                    else:
                        para_dict[key] = values[j]

                para_dict["fname"] = "{}-K100-E{}-Label2-{}-{}.log".format(
                    dataset, local_epochs, net, lr
                )

                main_federated(para_dict)


if __name__ == "__main__":
    # set seed
    setup_seed(seed=0)

    algos = [
        "fedavg", "fedprox", "fedmmd", "scaffold",
        "fedopt", "fednova", "fedaws", "moon",
        "perfedavg", "pfedme",
        "fedrs", "scaffoldrs", "fedphp",
    ]

    algos = [
        "scaffoldrs",
        "fedprox", "fedmmd", "fednova",
        "fedaws", "moon",
        "perfedavg", "pfedme",
    ]

    for dataset in ["cifar100"]:
        for algo in algos:
            main_cifar_label(dataset, algo)
