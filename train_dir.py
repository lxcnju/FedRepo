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
from algorithms.moon import MOON
from algorithms.feddyn import FedDyn

from algorithms.pfedme import pFedMe
from algorithms.perfedavg import PerFedAvg

from algorithms.fedphp import FedPHP

from networks.basic_nets import VGG

from paths import save_dir
from config import default_param_dicts

from utils import weights_init
from utils import setup_seed

torch.set_default_tensor_type(torch.FloatTensor)


def construct_model(args):
    model = VGG(
        n_layer=args.n_layer,
        n_classes=args.n_classes,
        use_bn=False
    )

    model.apply(weights_init)
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
    elif args.algo == "moon":
        FedAlgo = MOON
    elif args.algo == "feddyn":
        FedAlgo = FedDyn
    elif args.algo == "pfedme":
        FedAlgo = pFedMe
    elif args.algo == "perfedavg":
        FedAlgo = PerFedAvg
    elif args.algo == "fedphp":
        FedAlgo = FedPHP
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
            "cnt": 5,
            "reg_way": ["fedprox"] * 5,
            "reg_lamb": [1e-5, 1e-1, 1e-4, 1e-3, 1e-2]
        }
    elif algo == "fedmmd":
        hypers = {
            "cnt": 4,
            "reg_way": ["fedmmd"] * 4,
            "reg_lamb": [1e-2, 1e-3, 1e-4, 1e-1]
        }
    elif algo == "scaffold":
        hypers = {
            "cnt": 2,
            "glo_lr": [0.25, 0.5]
        }
    elif algo == "fedopt":
        hypers = {
            "cnt": 8,
            "glo_optimizer": [
                "SGD", "Adam", "SGD", "SGD", "Adam", "SGD", "SGD", "Adam"
            ],
            "glo_lr": [0.1, 3e-4, 0.05, 0.01, 1e-4, 0.3, 0.03, 5e-5],
        }
    elif algo == "fednova":
        hypers = {
            "cnt": 8,
            "gmf": [0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 0.75, 0.9],
            "prox_mu": [1e-3, 1e-3, 1e-4, 1e-2, 1e-4, 1e-5, 1e-4, 1e-3],
        }
    elif algo == "moon":
        hypers = {
            "cnt": 8,
            "reg_lamb": [1e-4, 1e-2, 1e-3, 1e-1, 1e-5, 1.0, 5e-4, 5e-3]
        }
    elif algo == "feddyn":
        hypers = {
            "cnt": 8,
            "reg_lamb": [1e-3, 1e-2, 1e-4, 1e-1, 1e-5, 1e-7, 1e-6, 5e-5]
        }
    elif algo == "pfedme":
        hypers = {
            "cnt": 8,
            "reg_lamb": [1e-4, 1e-2, 1e-3, 1e-5, 1e-4, 1e-5, 1e-5, 1e-4],
            "alpha": [0.1, 0.75, 0.5, 0.25, 0.5, 1.0, 0.75, 0.9],
            "k_step": [20, 10, 20, 20, 10, 5, 5, 10],
            "beta": [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0],
        }
    elif algo == "perfedavg":
        hypers = {
            "cnt": 5,
            "meta_lr": [0.05, 0.01, 0.1, 0.03, 0.005],
        }
    elif algo == "fedphp":
        hypers = {
            "cnt": 3,
            "reg_way": ["KD", "MMD", "MMD"],
            "reg_lamb": [0.1, 0.1, 0.05],
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


def main_cifar_dir(dataset, algo):
    hypers = get_hypers(algo)

    lr = 0.03
    for dir_alpha in [10.0, 1.0, 0.5, 0.1]:
        for j in range(hypers["cnt"]):
            para_dict = {}
            for k, vs in default_param_dicts[dataset].items():
                para_dict[k] = random.choice(vs)

            para_dict["algo"] = algo
            para_dict["dataset"] = dataset
            para_dict["n_layer"] = 8
            para_dict["split"] = "dirichlet"
            para_dict["dir_alpha"] = dir_alpha
            para_dict["lr"] = lr
            para_dict["n_clients"] = 100
            para_dict["c_ratio"] = 0.1
            para_dict["local_epochs"] = 5
            para_dict["max_round"] = 1000
            para_dict["test_round"] = 10

            for key, values in hypers.items():
                if key == "cnt":
                    continue
                else:
                    para_dict[key] = values[j]

            para_dict["fname"] = "{}-K100-Dir-{}-VGG8.log".format(
                dataset, dir_alpha
            )

            main_federated(para_dict)


if __name__ == "__main__":
    # set seed
    setup_seed(seed=0)

    algos = [
        "fedavg", "fedprox", "fedmmd", "scaffold",
        "fedopt", "fednova", "moon", "feddyn",
        "perfedavg", "pfedme", "fedphp",
    ]

    for dataset in ["cifar10"]:
        for algo in algos:
            main_cifar_dir(dataset, algo)
