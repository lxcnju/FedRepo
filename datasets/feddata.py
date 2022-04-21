import numpy as np

from collections import Counter

from datasets.famnist_data import load_famnist_data, FaMnistDataset
from datasets.cifar_data import load_cifar_data, CifarDataset

np.random.seed(0)


class FedData():
    """ Federated Datasets: support different scenes and split ways
    params:
    @dataset: "famnist", "cifar10", "cifar100"
    @split: "label", "user", None
        if split by "user", split each user to a client;
        if split by "label", split to n_clients with samples from several class
    @n_clients: int, None
        if split by "user", is Num.users;
        if split by "label", it is pre-defined;
    @nc_per_client: int, None
        number of classes per client, only for split="label";
    @n_client_perc: int, None
        number of clients per class, only for split="label" and dataset="sa";
    @dir_alpha: float > 0.0, 1.0
        Dir(alpha), the larger the uniform, 0.1, 1.0, 10.0
    @n_max_sam: int, None
        max number of samples per client, for low-resource learning;
    @split_sent140_way: str
        the way to split sent140
    """

    def __init__(
        self,
        dataset="mnist",
        test_ratio=0.2,
        split=None,
        n_clients=None,
        nc_per_client=None,
        n_client_perc=None,
        dir_alpha=1.0,
        n_max_sam=None,
    ):
        self.dataset = dataset
        self.test_ratio = test_ratio
        self.split = split
        self.n_clients = n_clients
        self.nc_per_client = nc_per_client
        self.n_client_perc = n_client_perc
        self.dir_alpha = dir_alpha
        self.n_max_sam = n_max_sam

        self.label_dsets = [
            "famnist",
            "cifar10", "cifar100"
        ]

        if dataset in self.label_dsets:
            assert self.split in ["label", "dirichlet"]

            assert (n_clients is not None), \
                "{} needs pre-defined n_clients".format(dataset)

            if self.split == "label":
                if dataset == "sa":
                    assert (n_client_perc is not None), \
                        "{} needs pre-defined n_client_perc".format(dataset)
                else:
                    assert (nc_per_client is not None), \
                        "{} needs pre-defined nc_per_client".format(dataset)

    def split_by_dirichlet(self, xs, ys):
        """ split data into N clients with distribution with Dir(alpha)
        params:
        @xs: numpy.array, shape=(N, ...)
        @ys: numpy.array, shape=(N, ), only for classes
        return:
        @clients_data, a dict like {
            client: {
                "train_xs":,
                "train_ys":,
                "test_xs":,
                "test_ys":
            }
        }
        """
        # unique classes
        n_classes = len(np.unique(ys))
        class_cnts = np.array([
            np.sum(ys == c) for c in range(n_classes)
        ])
        class_indxes = {
            c: np.argwhere(ys == c).reshape(-1) for c in range(n_classes)
        }

        # (n_clients, n_classes)
        dists = np.random.dirichlet(
            alpha=[self.dir_alpha] * n_classes,
            size=self.n_clients
        )
        dists = dists / dists.sum(axis=0)

        # (n_clients, n_classes)
        cnts = (dists * class_cnts.reshape((1, -1)))
        cnts = np.round(cnts).astype(np.int32)

        cnts = np.cumsum(cnts, axis=0)
        cnts = np.concatenate([
            np.zeros((1, n_classes)).astype(np.int32),
            cnts
        ], axis=0)

        # split data by Dists
        clients_data = {}
        for n in range(self.n_clients):
            client_xs = []
            client_ys = []
            for c in range(n_classes):
                cinds = class_indxes[c]
                bi, ei = cnts[n][c], cnts[n + 1][c]
                c_xs = xs[cinds[bi:ei]]
                c_ys = ys[cinds[bi:ei]]

                client_xs.append(c_xs)
                client_ys.append(c_ys)
                if n == self.n_clients - 1:
                    print(c, len(cinds), bi, ei)

            client_xs = np.concatenate(client_xs, axis=0)
            client_ys = np.concatenate(client_ys, axis=0)

            inds = np.random.permutation(client_xs.shape[0])
            client_xs = client_xs[inds]
            client_ys = client_ys[inds]

            # filter small corpus
            if len(client_xs) < 5:
                continue

            # split train and test
            n_test = max(int(self.test_ratio * len(client_xs)), 1)

            # max train samples
            if self.n_max_sam is None:
                n_end = None
            else:
                n_end = self.n_max_sam + n_test

            clients_data[n] = {
                "train_xs": client_xs[n_test:n_end],
                "train_ys": client_ys[n_test:n_end],
                "test_xs": client_xs[:n_test],
                "test_ys": client_ys[:n_test],
            }

        return clients_data

    def split_by_label(self, xs, ys):
        """ split data into N clients, each client has C classes
        params:
        @xs: numpy.array, shape=(N, ...)
        @ys: numpy.array, shape=(N, ), only for classes
        return:
        @clients_data, a dict like {
            client: {
                "train_xs":,
                "train_ys":,
                "test_xs":,
                "test_ys":
            }
        }
        """
        # unique classes
        uni_classes = sorted(np.unique(ys))
        seq_classes = []
        for _ in range(self.n_clients):
            np.random.shuffle(uni_classes)
            seq_classes.extend(list(uni_classes))

        # each class at least assigned to a client
        assert (self.nc_per_client * self.n_clients >= len(uni_classes)), \
            "Each class as least assigned to a client"

        # assign classes to each client
        client_classes = {}
        for k, client in enumerate(range(self.n_clients)):
            client_classes[client] = seq_classes[
                k * self.nc_per_client: (k + 1) * self.nc_per_client
            ]

        # for a class, how many clients have it
        classes = []
        for client in client_classes.keys():
            classes.extend(client_classes[client])

        classes_cnt = dict(Counter(classes))

        # shuffle xs, and ys
        n_samples = xs.shape[0]
        inds = np.random.permutation(n_samples)
        xs = xs[inds]
        ys = ys[inds]

        # assign classes to each client
        clients_data = {}
        for client in client_classes.keys():
            clients_data[client] = {
                "xs": [],
                "ys": []
            }

        # split data by classes
        for c in uni_classes:
            cinds = np.argwhere(ys == c).reshape(-1)
            c_xs = xs[cinds]
            c_ys = ys[cinds]

            # assign class data uniformly to each client
            t = 0
            for client, client_cs in client_classes.items():
                if c in client_cs:
                    n = client_cs.count(c)
                    ind1 = t * int(len(c_xs) / classes_cnt[c])
                    ind2 = (t + n) * int(len(c_xs) / classes_cnt[c])
                    clients_data[client]["xs"].append(c_xs[ind1:ind2])
                    clients_data[client]["ys"].append(c_ys[ind1:ind2])
                    t += n
            assert (t == classes_cnt[c]), \
                "Error, t != classes_cnt[c]"

        # shuffle data and limit maximum number
        for client, values in clients_data.items():
            client_xs = np.concatenate(values["xs"], axis=0)
            client_ys = np.concatenate(values["ys"], axis=0)

            inds = np.random.permutation(client_xs.shape[0])
            client_xs = client_xs[inds]
            client_ys = client_ys[inds]

            # filter small corpus
            if len(client_xs) < 5:
                continue

            # split train and test
            n_test = max(int(self.test_ratio * len(client_xs)), 1)

            # max train samples
            if self.n_max_sam is None:
                n_end = None
            else:
                n_end = self.n_max_sam + n_test

            clients_data[client] = {
                "train_xs": client_xs[n_test:n_end],
                "train_ys": client_ys[n_test:n_end],
                "test_xs": client_xs[:n_test],
                "test_ys": client_ys[:n_test],
            }

        return clients_data

    def construct_datasets(
            self, clients_data, Dataset, glo_test_xs=None, glo_test_ys=None):
        """
        params:
        @clients_data, {
            client: {
                "train_xs":,
                "train_ys":,
                "test_xs":,
                "test_ys":
            }
        }
        @Dataset: torch.utils.data.Dataset type
        @glo_test_xs: global test xs, ys
        @glo_test_ys: global test xs, ys
        return: client train and test Datasets and global test Dataset
        @csets: {
            client: (train_set, test_set)
        }
        @gset: data.Dataset
        """
        csets = {}

        if glo_test_xs is None or glo_test_ys is None:
            glo_test = False
        else:
            glo_test = True

        if glo_test is False:
            glo_test_xs = []
            glo_test_ys = []

        for client, cdata in clients_data.items():
            train_set = Dataset(
                cdata["train_xs"], cdata["train_ys"], is_train=True
            )
            test_set = Dataset(
                cdata["test_xs"], cdata["test_ys"], is_train=False
            )
            csets[client] = (train_set, test_set)

            if glo_test is False:
                glo_test_xs.append(cdata["test_xs"])
                glo_test_ys.append(cdata["test_ys"])

        if glo_test is False:
            glo_test_xs = np.concatenate(glo_test_xs, axis=0)
            glo_test_ys = np.concatenate(glo_test_ys, axis=0)

        gset = Dataset(glo_test_xs, glo_test_ys, is_train=False)
        return csets, gset

    def split_by_label_noniid(self, xs, ys):
        if self.split == "label":
            clients_data = self.split_by_label(xs, ys)
        elif self.split == "dirichlet":
            clients_data = self.split_by_dirichlet(xs, ys)
        else:
            raise ValueError("No such split: {}".format(self.split))
        return clients_data

    def construct(self):
        """ load raw data
        """
        if self.dataset == "famnist":
            train_xs, train_ys, test_xs, test_ys = load_famnist_data(
                "famnist", combine=False
            )
            clients_data = self.split_by_label_noniid(train_xs, train_ys)
            csets, gset = self.construct_datasets(
                clients_data, FaMnistDataset, test_xs, test_ys
            )
        elif self.dataset in ["cifar10", "cifar100"]:
            train_xs, train_ys, test_xs, test_ys = load_cifar_data(
                self.dataset, combine=False
            )
            clients_data = self.split_by_label_noniid(train_xs, train_ys)
            csets, gset = self.construct_datasets(
                clients_data, CifarDataset, test_xs, test_ys
            )
        else:
            raise ValueError("No such dataset: {}".format(self.dataset))

        return csets, gset

    def print_info(self, csets, gset, max_cnt=5):
        """ print information
        """
        print("#" * 50)
        cnt = 0
        print("Dataset:{}".format(self.dataset))
        print("N clients:{}".format(len(csets)))

        for client, (cset1, cset2) in csets.items():
            print("Information of Client {}:".format(client))
            print(
                "Local Train Set: ", cset1.xs.shape,
                cset1.xs.max(), cset1.xs.min(), Counter(cset1.ys)
            )
            print(
                "Local Test Set: ", cset2.xs.shape,
                cset2.xs.max(), cset2.xs.min(), Counter(cset2.ys)
            )

            cnts = [n for _, n in Counter(cset1.ys).most_common()]
            probs = np.array([n / sum(cnts) for n in cnts])
            ent = -1.0 * (probs * np.log(probs + 1e-8)).sum()
            print("Class Distribution, Min:{}, Max:{}, Ent:{}".format(
                np.min(probs), np.max(probs), ent
            ))

            if cnt >= max_cnt:
                break
            cnt += 1

        print(
            "Global Test Set: ", gset.xs.shape,
            gset.xs.max(), gset.xs.min(), Counter(gset.ys)
        )
        print("#" * 50)
