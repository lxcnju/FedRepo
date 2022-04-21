import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders
from tools import construct_optimizer


# code link:
# https://github.com/alpemreacar/FedDyn


class FedDyn():
    def __init__(
        self,
        csets,
        gset,
        model,
        args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())
        self.n_client = len(self.clients)

        # build private gradients for each client
        self.client_grads = {}
        for client in self.clients:
            self.client_grads[client] = self.build_grad_dict(model)

        # global grad dict
        # self.glo_grad = self.build_grad_dict(model)

        # to cuda
        if self.args.cuda is True:
            self.model = self.model.cuda()
            # self.to_device(self.glo_grad, "gpu")

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

    def build_grad_dict(self, model):
        grad_dict = {}
        for key, params in model.state_dict().items():
            grad_dict[key] = torch.zeros_like(params)
        return grad_dict

    def to_device(self, grad_dict, device):
        for key in grad_dict.keys():
            if device == "gpu":
                grad_dict[key] = grad_dict[key].cuda()
            else:
                grad_dict[key] = grad_dict[key].cpu()

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
                # to cuda
                if self.args.cuda is True:
                    self.to_device(self.client_grads[client], "gpu")

                local_model, local_grad, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    local_grad=copy.deepcopy(self.client_grads[client]),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_models[client] = copy.deepcopy(local_model)

                # update local grad
                self.to_device(local_grad, "cpu")
                self.client_grads[client] = copy.deepcopy(local_grad)

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

    def update_local(self, r, model, local_grad, train_loader, test_loader):
        glo_model = copy.deepcopy(model)
        glo_model.eval()

        optimizer = construct_optimizer(
            model, self.args.lr, self.args
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

            hs, logits = model(batch_x)

            criterion = nn.CrossEntropyLoss()
            ce_loss = criterion(logits, batch_y)

            # FedDyn Loss
            reg_loss = 0.0
            cnt = 0.0
            for name, param in model.named_parameters():
                term1 = (param * (
                    local_grad[name] - glo_model.state_dict()[name]
                )).sum()
                term2 = (param * param).sum()

                reg_loss += self.args.reg_lamb * (term1 + term2)
                cnt += 1.0

            loss = ce_loss + reg_loss / cnt

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()

        # update local_grad
        for name, param in model.named_parameters():
            local_grad[name] += (
                model.state_dict()[name] - glo_model.state_dict()[name]
            )
        return model, local_grad, per_accs, loss

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

            alpha = self.args.c_ratio
            mean_state_dict[name] = alpha * mean_value + (1.0 - alpha) * param

        global_model.load_state_dict(mean_state_dict, strict=False)

    """
    def update_global(self, r, global_model, local_models, glo_grad):
        sum_state_dict = {}

        K = len(self.clients)
        M = int(self.args.c_ratio * K)

        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                sum_value = vs.sum(dim=0)
            except Exception:
                # for BN's cnt
                sum_value = (1.0 * vs).sum(dim=0).long()
            sum_state_dict[name] = sum_value

        # update glo grad & global model
        state_dict = {}
        lamb = self.args.reg_lamb
        for name, param in global_model.state_dict().items():
            glo_grad[name] -= lamb / K * (
                sum_state_dict[name] - param
            )

            state_dict[name] = sum_state_dict[name] / M - glo_grad[name] / lamb

            if "run" in name:
                state_dict[name] = state_dict[name].long()

        global_model.load_state_dict(state_dict, strict=False)
    """

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
