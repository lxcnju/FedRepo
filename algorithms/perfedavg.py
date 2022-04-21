import copy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders


# code link:
# https://github.com/CharlieDinh/pFedMe


class MySGD(Optimizer):
    def __init__(self, params, lr, mu, weight_decay):
        defaults = dict(lr=lr, mu=mu, weight_decay=weight_decay)
        super(MySGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, beta=0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                if group["weight_decay"] != 0.0:
                    d_p = d_p + group["weight_decay"] * p.data
                    # d_p = d_p.add(p, alpha=group["weight_decay"])

                if group["mu"] != 0.0:
                    if "momentum_buffer" not in param_state:
                        buf = torch.clone(d_p).detach()
                        param_state["momentum_buffer"] = buf
                    else:
                        buf = param_state["momentum_buffer"]
                        buf = group["mu"] * buf + d_p
                        # buf.mul_(group["mu"]).add_(d_p)

                        # update momentum buffer important !!!
                        param_state["momentum_buffer"] = buf
                    d_p = buf

                if (beta != 0):
                    p.data = p.data - beta * d_p
                else:
                    p.data = p.data - group['lr'] * d_p
        return loss


class PerFedAvg():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
        }

    def train(self):
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            local_models = {}

            avg_loss = Averager()
            all_per_accs = []
            for client in sam_clients:
                local_model, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_models[client] = copy.deepcopy(local_model)
                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
            )

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc = self.test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["LOCAL_TACCS"].extend(per_accs)

                print("[R:{}] [Ls:{}] [TeAc:{}] [PAcBeg:{} PAcAft:{}]".format(
                    r, train_loss, glo_test_acc, per_accs[0], per_accs[-1]
                ))

    def update_local(self, r, model, train_loader, test_loader):
        # lr = min(r / 10.0, 1.0) * self.args.lr
        lr = self.args.lr

        optimizer = MySGD(
            params=model.parameters(),
            lr=lr,
            mu=self.args.momentum,
            weight_decay=self.args.weight_decay
        )

        if self.args.local_steps is not None:
            n_total_bs = self.args.local_steps
        elif self.args.local_epochs is not None:
            n_total_bs = max(
                int(self.args.local_epochs * len(train_loader)), 5
            )
        else:
            raise ValueError(
                "local_steps and local_epochs must not be None together"
            )

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                per_acc = self.test(
                    model=model,
                    loader=test_loader,
                )
                per_accs.append(per_acc)

            if t >= n_total_bs:
                break

            model.train()
            try:
                batch_x, batch_y = loader_iter.next()
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = loader_iter.next()

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            temp_params = copy.deepcopy(list(model.parameters()))

            # step 1
            optimizer.zero_grad()
            _, logits = model(batch_x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            avg_loss.add(loss.item())

            # step 2
            try:
                batch_x, batch_y = loader_iter.next()
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = loader_iter.next()

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            _, logits = model(batch_x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)
            loss.backward()

            # restore the model parameters to the one before first update
            for p1, p0 in zip(model.parameters(), temp_params):
                p1.data = p0.data.clone()

            optimizer.step(beta=self.args.meta_lr)

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        return model, per_accs, loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, logits = model(batch_x)
                acc = count_acc(logits, batch_y)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)

