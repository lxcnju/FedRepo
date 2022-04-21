import torch
from torch.utils.data import DataLoader


def guassian_kernel(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    L2_distance = ((
        total.unsqueeze(dim=1) - total.unsqueeze(dim=0)
    ) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth += 1e-8

    # print("Bandwidth:", bandwidth)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / band) for band in bandwidth_list
    ]
    return sum(kernel_val)


def mmd_rbf_noaccelerate(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def construct_dataloaders(clients, csets, gset, args):
    train_loaders = {}
    test_loaders = {}
    glo_test_loader = None

    for client in clients:
        assert isinstance(csets[client], tuple), \
            "csets must be a tuple (train_set, test_set): {}".format(client)

        assert csets[client][1] is not None, \
            "local test set must not be None in client: {}".format(client)

        train_loader = DataLoader(
            csets[client][0],
            batch_size=args.batch_size,
            shuffle=True
        )
        train_loaders[client] = train_loader

        test_loader = DataLoader(
            csets[client][1],
            batch_size=args.batch_size * 10,
            shuffle=False
        )
        test_loaders[client] = test_loader

    assert gset is not None, \
        "global test set must not be None"

    glo_test_loader = DataLoader(
        gset,
        batch_size=args.batch_size * 10,
        shuffle=False
    )

    return train_loaders, test_loaders, glo_test_loader


def construct_optimizer(model, lr, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(
            args.optimizer
        ))
    return optimizer
