import copy
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders
from tools import construct_optimizer


class FedRS():
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
                dist = cnts / cnts.sum()

                local_model, per_accs, per_mf1s, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                    dist=dist
                )

                local_models[client] = copy.deepcopy(local_model)
                avg_loss.add(loss)

                all_per_accs.append(per_accs)
                all_per_mf1s.append(per_mf1s)
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

    def update_local(self, r, model, train_loader, test_loader, dist):
        # lr = min(r / 10.0, 1.0) * self.args.lr
        lr = self.args.lr

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
                per_acc, per_mf1 = self.test(
                    model=model,
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

            if self.args.cuda:
                dist = dist.cuda()

            hs, _ = model(batch_x)
            ws = model.classifier.weight

            cdist = dist / dist.max()
            cdist = cdist * (1.0 - self.args.alpha) + self.args.alpha
            cdist = cdist.reshape((1, -1))

            logits = cdist * hs.mm(ws.transpose(0, 1))

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
        return model, per_accs, per_mf1s, loss

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

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
