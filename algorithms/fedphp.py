import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders
from tools import construct_optimizer

from tools import mmd_rbf_noaccelerate


class FedPHP():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model.cpu()
        self.args = args

        self.clients = list(csets.keys())

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.hpms = {}
        for client in self.clients:
            self.hpms[client] = copy.deepcopy(self.model)

        # to cuda
        if self.args.cuda is True:
            self.model = self.model.cuda()

        self.cnts = {}
        for client in self.clients:
            self.cnts[client] = 0

        # client_cnts
        self.client_cnts = self.get_client_dists(
            csets=self.csets,
            args=self.args
        )

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
            "LOCAL_MF1S": [],
        }

    def get_client_dists(self, csets, args):
        client_cnts = {}
        for client in csets.keys():
            info = csets[client]

            cnts = [
                np.sum(info[0].ys == c) for c in range(args.n_classes)
            ]

            cnts = torch.FloatTensor(np.array(cnts))
            client_cnts[client] = cnts

        return client_cnts

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
            all_per_mf1s = []

            weights = {}
            total_cnts = 0.0
            for client in sam_clients:
                cnts = self.client_cnts[client]

                local_model, hpm, paccs, pmf1s, loss \
                    = self.update_local(
                        r=r,
                        client=client,
                        model=copy.deepcopy(self.model),
                        hpm=self.hpms[client],
                        train_loader=self.train_loaders[client],
                        test_loader=self.test_loaders[client],
                    )

                # cnts
                self.cnts[client] += 1

                local_models[client] = copy.deepcopy(local_model)
                self.hpms[client] = copy.deepcopy(hpm)

                avg_loss.add(loss)
                all_per_accs.append(paccs)
                all_per_mf1s.append(pmf1s)

                weights[client] = cnts.sum()
                total_cnts += cnts.sum()

            weights = {k: v / total_cnts for k, v in weights.items()}

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))
            per_mf1s = list(np.array(all_per_mf1s).mean(axis=0))

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
                weights=weights
            )

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc, _ = self.test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["LOCAL_TACCS"].extend(per_accs)
                self.logs["LOCAL_MF1S"].extend(per_mf1s)

                print("[R:{}] [Ls:{}] [TAc:{}] [PAc:{},{}] [PF1:{},{}]".format(
                    r, train_loss, glo_test_acc, per_accs[0], per_accs[-1],
                    per_mf1s[0], per_mf1s[-1]
                ))

    def update_local(self, r, client, model, hpm, train_loader, test_loader):
        # lr = min(r / 10.0, 1.0) * self.args.lr
        lr = self.args.lr

        if self.args.cuda is True:
            hpm = hpm.cuda()

        optimizer = construct_optimizer(
            model, lr, self.args
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
        per_mf1s = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                """
                per_acc, per_mf1 = self.local_test(
                    model=model,
                    hpm=hpm,
                    loader=test_loader,
                )
                """
                per_acc, per_mf1 = self.test(
                    model=hpm,
                    loader=test_loader,
                )
                per_accs.append(per_acc)
                per_mf1s.append(per_mf1)

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
            phs, plogits = hpm(batch_x)
            phs = phs.detach()
            plogits = plogits.detach()

            # kl loss
            criterion = nn.CrossEntropyLoss()
            ce_loss = criterion(logits, batch_y)

            # knowledge transfer
            if self.args.reg_way == "KD":
                reg_loss = (
                    -1.0 * (plogits / 4.0).softmax(
                        dim=1
                    ) * logits.log_softmax(dim=1)
                ).sum(dim=1).mean()
            elif self.args.reg_way == "MMD":
                reg_loss = mmd_rbf_noaccelerate(hs, phs)
            else:
                raise ValueError("No such reg way: {}".format(
                    self.args.reg_way
                ))

            coef = self.args.reg_lamb
            loss = (1.0 - coef) * ce_loss + coef * reg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

            if t >= n_total_bs - 1:
                hpm = self.update_hpm(client, model, hpm)

        loss = avg_loss.item()

        # update hpm
        hpm = hpm.cpu()

        return model, hpm, per_accs, per_mf1s, loss

    def generate_mu(self, client):
        cnt = self.cnts[client]
        mean_sam = max(int(self.args.c_ratio * self.args.max_round), 1)
        mu = 0.9 * cnt / mean_sam
        mu = min(max(mu, 0.0), 0.9)
        return mu

    def update_hpm(self, client, model, hpm):
        mu = self.generate_mu(client)
        mean_state_dict = {}
        for name, p_param in hpm.state_dict().items():
            s_param = model.state_dict()[name]
            mean_state_dict[name] = mu * p_param + (1.0 - mu) * s_param

        hpm.load_state_dict(
            mean_state_dict, strict=False
        )
        return hpm

    def update_global(self, r, global_model, local_models, weights):
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                w = weights[client]
                vs.append(w * local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.sum(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).sum(dim=0).long()
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        preds = []
        reals = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, logits = model(batch_x)
                acc = count_acc(logits, batch_y)
                acc_avg.add(acc)

                preds.append(np.argmax(logits.cpu().detach().numpy(), axis=1))
                reals.append(batch_y.cpu().detach().numpy())

        preds = np.concatenate(preds, axis=0)
        reals = np.concatenate(reals, axis=0)

        acc = acc_avg.item()

        # MACRO F1
        mf1 = f1_score(y_true=reals, y_pred=preds, average="macro")
        return acc, mf1

    def local_test(self, model, hpm, loader):
        model.eval()
        hpm.eval()

        acc_avg = Averager()

        preds = []
        reals = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                _, logits = model(batch_x)
                _, plogits = hpm(batch_x)

                probs = 0.5 * (
                    logits.softmax(dim=-1) + plogits.softmax(dim=-1)
                )
                acc = count_acc(probs, batch_y)

                acc_avg.add(acc)

                preds.append(np.argmax(probs.cpu().detach().numpy(), axis=1))
                reals.append(batch_y.cpu().detach().numpy())

        preds = np.concatenate(preds, axis=0)
        reals = np.concatenate(reals, axis=0)

        acc = acc_avg.item()

        # MACRO F1
        mf1 = f1_score(y_true=reals, y_pred=preds, average="macro")
        return acc, mf1

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
