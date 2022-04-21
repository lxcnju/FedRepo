import copy
import numpy as np

import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders

# https://github.com/JYWa/FedNova


class NovaOptimizer(Optimizer):
    """ gmf: global momentum
        prox_mu: mu of proximal term
        ratio: client weight
    """

    def __init__(
        self, params, lr, ratio, gmf, prox_mu=0,
        momentum=0, dampening=0, weight_decay=0, nesterov=False, variance=0
    ):
        self.gmf = gmf
        self.ratio = ratio
        self.prox_mu = prox_mu
        self.momentum = momentum
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if lr < 0.0:
            raise ValueError("Invalid lr: {}".format(lr))

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov, variance=variance
        )
        super(NovaOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NovaOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                # weight_decay
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # save the first parameter w0
                param_state = self.state[p]
                if "old_init" not in param_state:
                    param_state["old_init"] = torch.clone(p.data).detach()

                # momentum:
                # v_{t+1} = rho * v_t + g_t
                # g_t = v_{t+1}
                # rho = momentum
                local_lr = group["lr"]
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = torch.clone(d_p).detach()
                        param_state["momentum_buffer"] = buf
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)

                        # update momentum buffer !!!
                        param_state["momentum_buffer"] = buf

                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # add proximal updates: g_t = g_t + prox_mu * (w - w0)
                if self.prox_mu != 0:
                    d_p.add_(self.prox_mu, p.data - param_state["old_init"])

                # updata accumulated local updates
                # sum(g_0, g_1, ..., g_t)
                if "cum_grad" not in param_state:
                    param_state["cum_grad"] = torch.clone(d_p).detach()
                    param_state["cum_grad"].mul_(local_lr)
                else:
                    param_state["cum_grad"].add_(local_lr, d_p)

                # update: w_{t+1} = w_t - lr * g_t
                p.data.add_(-1.0 * local_lr, d_p)

        # compute local normalizing vec, a_i
        # For momentum: a_i = [(1 - rho)^{tau_i - 1}/(1 - rho), ..., 1]
        # 1, 1 + rho, 1 + rho + rho^2, ...
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        # proximal: a_i = [(1 - eta * mu)^{\tau_i - 1}, ..., 1]
        # 1, 1 - eta * mu, (1 - eta * mu)^2 + 1, ...
        self.etamu = local_lr * self.prox_mu
        if self.etamu != 0:
            self.local_normalizing_vec *= (1 - self.etamu)
            self.local_normalizing_vec += 1

        # FedAvg: no momentum, no proximal, [1, 1, 1, ...]
        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1
        return loss


class FedNova():
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

        self.global_momentum_buffer = {}

    def train(self):
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            avg_loss = Averager()
            all_per_accs = []

            local_grads = []
            tau_effs = []

            for client in sam_clients:
                local_grad, tau_eff, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                    ratio=1.0 / n_sam_clients,
                )

                local_grads.append(copy.deepcopy(local_grad))
                tau_effs.append(tau_eff)

                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            self.update_global(
                r=r,
                global_model=self.model,
                local_grads=local_grads,
                tau_effs=tau_effs,
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

    def get_local_grad_(self, opt, cur_params, init_params):
        weight = opt.ratio

        grad_dict = {}
        for k in cur_params.keys():
            scale = 1.0 / opt.local_normalizing_vec
            cum_grad = init_params[k] - cur_params[k]
            try:
                cum_grad.mul_(weight * scale)
            except Exception:
                cum_grad = (cum_grad * weight * scale).long()
            grad_dict[k] = cum_grad
        return grad_dict

    def update_local(self, r, model, train_loader, test_loader, ratio):
        init_state_dict = copy.deepcopy(model.state_dict())

        optimizer = NovaOptimizer(
            model.parameters(),
            lr=self.args.lr,
            gmf=self.args.gmf,
            prox_mu=self.args.prox_mu,
            ratio=ratio,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
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
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()

        local_grad = self.get_local_grad_(
            opt=optimizer,
            cur_params=model.state_dict(),
            init_params=init_state_dict
        )

        if self.args.prox_mu != 0:
            tau_eff = optimizer.local_steps * optimizer.ratio
        else:
            tau_eff = optimizer.local_normalizing_vec * optimizer.ratio

        return local_grad, tau_eff, per_accs, loss

    def update_global(self, r, global_model, local_grads, tau_effs):
        tau_eff = sum(tau_effs)
        params = global_model.state_dict()

        cum_grad = local_grads[0]
        for k in local_grads[0].keys():
            for i in range(0, len(local_grads)):
                if i == 0:
                    cum_grad[k] = local_grads[i][k] * tau_eff
                else:
                    cum_grad[k] += local_grads[i][k] * tau_eff

        for k in params.keys():
            if self.args.gmf != 0:
                if k not in self.global_momentum_buffer:
                    self.global_momentum_buffer[k] = torch.clone(
                        cum_grad[k]
                    ).detach()
                    buf = self.global_momentum_buffer[k]
                    buf.div_(self.args.lr)
                else:
                    buf = self.global_momentum_buffer[k]
                    buf.mul_(self.args.gmf).add_(
                        1.0 / self.args.lr, cum_grad[k]
                    )
                try:
                    params[k].sub_(self.args.lr, buf)
                except Exception:
                    params[k] = (params[k] - self.args.lr * buf).long()
            else:
                try:
                    params[k].sub_(cum_grad[k])
                except Exception:
                    params[k] = (params[k] - cum_grad[k]).long()

        global_model.load_state_dict(params, strict=True)

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
