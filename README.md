# FedRepo
The source code of our works on federated learning:
* KDD 2021 paper: FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data.
* ECML/PKDD 2021 paper: FedPHP: Federated Personalization with Inherited Private Models.
* CVPR 2022 paper: Federated Learning with Position-Aware Neurons. \[**Source Code in [FedPAN](https://github.com/lxcnju/FedPAN/)**\]


# Content
* Personal Homepage
* Basic Introduction
* Code Files
* Running Tips
* Citation

## Personal Homepage
  * [Homepage](https://www.lamda.nju.edu.cn/lixc/)

## Basic Introduction
  * Federated Learning (FL) fuses collaborative models from local nodes without centralizing users' data.
  * We implement several popular FL methods in recent years and provide source code of our works.

## Implemented FL Algorithms
We implement several popular FL algorithms with local regularization (e.g., FedProx, FedDyn), better optimization (e.g., FedNova, FedOpt), control variates (e.g., Scaffold), contrastive learning (e.g., MOON), etc. FedRS and FedPHP are our proposed methods. Some methods are for better aggregation (e.g., FedOpt, Scaffold, FedRS), while some are for better personalization (e.g., pFedMe, PerFedAvg, FedPHP). More about FL aggregation and personalization could be found in the technical report: [**Aggregate or Not**](https://arxiv.org/abs/2107.11954).
  * \[**FedAvg**\] Communication-efficient learning of deep networks from decentralized data. AISTATS 2017.
  * \[**FedProx**\] Federated optimization in heterogeneous networks. MLSys 2020.
  * \[**FedMMD**\] Two-Stream Federated Learning: Reduce the Communication Costs. VCIP 2018.
  * \[**FedNova**\] Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization. NeurIPS 2020.
  * \[**FedAwS**\] Federated Learning with Only Positive Labels. ICML 2020.
  * \[**MOON**\] Model contrastive federated learning. CVPR 2021.
  * \[**FedOpt**\] Adaptive federated optimization. ICLR 2021.
  * \[**Scaffold**\] SCAFFOLD: stochastic controlled averaging for federated learning. ICML 2020.
  * \[**pFedMe**\] Personalized Federated Learning with Moreau Envelopes. NeurIPS 2020.
  * \[**PerFedAvg**\] Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach. NeurIPS 2020.
  * \[**FedDyn**\] Federated Learning Based on Dynamic Regularization. ICLR 2021.
  * \[**FedRS** & **ScaffoldRS**\] FedRS: Federated learning with restricted softmax for label distribution non-iid data. KDD 2021.
  * \[**FedPHP**\] FedPHP: Federated personalization with inherited private models. In ECML/PKDD 2021.

These algorithms could be found in the directory of `algorithms/'.

## Code References During Implementation
  * [FedDyn](https://github.com/alpemreacar/FedDyn)
  * [FedNova](https://github.com/JYWa/FedNova)
  * [pFedMe & PerFedAvg](https://github.com/CharlieDinh/pFedMe)
  * [Scaffold](https://github.com/ramshi236/Accelerated-Federated-Learning-Over-MAC-in-Heterogeneous-Networks)
  * [MOON](https://github.com/QinbinLi/MOON)

## Environment Dependencies
The code files are written in Python, and the utilized deep learning tool is PyTorch.
  * `python`: 3.7.3
  * `numpy`: 1.21.5
  * `torch`: 1.9.0
  * `torchvision`: 0.10.0
  * `pillow`: 8.3.1

## Datasets
We provide several datasets including (downloading link code be found in my [Homepage](https://www.lamda.nju.edu.cn/lixc/)):
  * FaMnist
  * CIFAR-10
  * CIFAR-100

We provide several FL dataset constructing ways including:
  * **Split By Label**: each client owns *nc_per_client* classes, e.g., *nc_per_client=2* for CIFAR-10;
  * **Split By Dirichlet**: each client owns a label distribution sampled from Dirichlet distribution, *dir_alpha* determines the non-i.i.d. level.

These codes are encapsulated into the `datasets/feddata.py`.


## Running Tips
  * `python train_lab.py`: run algorithms under the *Split By Label* scene with *nc_per_client=2* for CIFAR-10;
  * `python train_lab_scenes.py`: run several algorithms under the *Split By Label* scene with *nc_per_client* in {3, 4, 5} for CIFAR-10;
  * `python train_dir.py`: run algorithms under the *Split By Dirichlet* scene.

FL algorithms and hyper-parameters could be set in these files.


## Citation
  * Xin-Chun Li, Yi-Chu Xu, Shaoming Song, Bingshuai Li, Yinchuan Li, Yunfeng Shao, De-Chuan Zhan. Federated Learning with Position-Aware Neurons. In: Proceedings of the 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR'2022), online conference, New Orleans, Louisiana, 2022.
  * Xin-Chun Li, De-Chuan Zhan, Yunfeng Shao, Bingshuai Li, Shaoming Song. FedPHP: Federated Personalization with Inherited Private Models. In: Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD'21), online conference, Bilbao, Spain, 2021.
  * Xin-Chun Li, De-Chuan Zhan. FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data. In: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'21), online conference, Singapore, 2021.
  * \[[BibTex](https://dblp.org/pid/246/2947.html)\]
