import os
from collections import Counter

import torch.nn.utils.prune as prune

import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from torch.autograd import Variable
from tqdm import tqdm

from GCL.models import DualBranchContrast
import GCL.losses as L
import GCL.augmentors as A

from torch.nn import Linear, ReLU
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, to_dense_batch

from CGT_model.CGT import CGT
from aug.DropPath import DropPath
import random
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import warnings

from aug.gumble import gumbel_softmax
from util import save_accs, gene_arg

warnings.filterwarnings("ignore")
now = datetime.now()
now = now.strftime("%m_%d-%H_%M_%S")

transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

args = gene_arg()


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def svc_classify(fold, x, y, search):
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in tqdm(kf.split(x, y), total=kf.get_n_splits()):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies), np.std(accuracies)


def load_data(args, transform):
    data_name = args.dataset + '-pe'
    dataset = TUDataset(os.path.join(args.data_root, data_name),
                        name=args.dataset
                        )
    if dataset.data.x is None:
        if "REDDIT" not in args.dataset:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)
        else:
            feature_dim = 0
            degrees = []
            for g in dataset:
                feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())
                degrees.extend(degree(g.edge_index[0]).tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for i, g in enumerate(dataset):
                degrees = degree(g.edge_index[0])
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
                feat = F.one_hot(degrees.to(torch.long), num_classes=int(feature_dim)).float()
                g.x = feat
                dataset[i] = g

    num_tasks = dataset.num_classes
    num_features = dataset.num_features
    num_dataset = len(dataset)
    all_loader = DataLoader(dataset, batch_size=args.batch_size)

    return dataset, all_loader, num_tasks, num_features, num_dataset


datasets, all_loader, num_tasks, num_features, num_dataset = load_data(args, transform)

device = torch.device('cuda:{}'.format(args.devices) if torch.cuda.is_available() else 'cpu')

model = CGT(fea_dim=num_features, channels=args.channels, num_layers=args.num_layers,
            num_tasks=num_tasks, num_features=num_features).to(device)
model_new = CGT(fea_dim=num_features, channels=args.channels, num_layers=args.num_layers,
                num_tasks=num_tasks, num_features=num_features).to(device)
contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

# MLP
projector = torch.nn.Sequential(
    Linear(args.channels, args.channels),
    ReLU(inplace=True),
    Linear(args.channels, args.channels)).to(device)


def data_aug(x, batch, pe, edge_index, method):
    x1, edge_index1 = x, edge_index
    x2, edge_index2 = x, edge_index
    ratio = args.aug_ratio
    if method == "TokenMask":
        h, mask = to_dense_batch(x, batch)
        B = h.shape[0]
        node_num = h.shape[1]
        features = h.shape[2]
        node_score = model.gumbel(h)
        node_score = node_score.reshape(B, node_num)
        node_mask = gumbel_softmax(node_score, device=device, rate=1 - ratio, tau=1, hard=True)
        node_mask[:, 0] = 1.
        node_mask = node_mask.expand(features, -1, -1).permute(1, 2, 0)
        h = h * node_mask
        x2 = h[mask]
    elif method == "FeatureMask":
        aug1 = A.FeatureMasking(pf=ratio)
        aug2 = A.FeatureMasking(pf=ratio)
        x1, edge_index1, _ = aug1(x, edge_index)
        x2, edge_index2, _ = aug2(x, edge_index)
    elif method == 'PEMask':
        mask_num = int(pe.size(1) * ratio)
        mask_col = random.sample(range(pe.size(1)), mask_num)
        mask = torch.ones_like(pe)
        mask[:, mask_col] = 0
        pe = pe * mask
    elif method == 'MAE':
        pretrain_aug = torch.load(f"mae_model/{args.dataset}.pt")
        pretrain_aug.to(device)
        pretrain_aug.eval()
        _, _, gene_x = pretrain_aug(x1, edge_index1)
        num_nodes = gene_x.shape[0] * 0.2
        ran_idx = torch.randperm(gene_x.shape[0])[:int(num_nodes)]
        x1[ran_idx] = gene_x[ran_idx]

    return x1, edge_index1, x2, edge_index2, pe


def model_aug(method):
    ratio = args.aug_ratio
    model_new.load_state_dict(model.state_dict())
    if method == "Gaussian":
        std = 0.01
        mean = 0.0
        for param in model_new.parameters():
            if param.requires_grad:
                noise = torch.normal(mean=mean, std=std, size=param.size()).to(device)
                param.data.add_(noise)
    elif method == "DropWeight":
        parameters_to_prune = []
        for name, module in model_new.named_modules():
            if not list(module.children()):
                for param_name, param in module.named_parameters():
                    if param_name == "weight":
                        parameters_to_prune.append((module, param_name))
        for module, param_name in parameters_to_prune:
            prune.l1_unstructured(module, param_name, amount=ratio)
            prune.remove(module, param_name)
    elif method == "DropPath":
        model_new.drop_path_prob = ratio
        model_new.drop_path = DropPath(model_new.drop_path_prob)
    elif method == "DropHead":
        for module in model_new.convs:
            module.attn_drophead = ratio


def train_data(loader, data_method, model_method):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        x1, edge_index1, x2, edge_index2, pe = data_aug(data.x, data.batch, data.pe, data.edge_index, data_method)
        _, g1, g2 = model(data.x, x1, x2, edge_index1, edge_index2, pe, data.edge_index, data.edge_attr,
                          data.batch)
        g1, g2 = [projector(g) for g in [g1, g2]]
        train_loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        train_loss.backward()
        total_loss += train_loss.item()
        optimizer.step()
    return total_loss


def train_model(loader, data_method, model_method):
    model.train()
    total_loss = 0
    model_aug(model_method)
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        g1, _, _ = model(data.x, data.x, data.x, data.edge_index, data.edge_index, data.pe, data.edge_index,
                         data.edge_attr, data.batch)
        g2, _, _ = model_new(data.x, data.x, data.x, data.edge_index, data.edge_index, data.pe, data.edge_index,
                             data.edge_attr, data.batch)
        g1, g2 = [projector(g) for g in [g1, g2]]
        g2 = Variable(g2.detach().data, requires_grad=False)
        train_loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        train_loss.backward()
        total_loss += train_loss.item()
        optimizer.step()
    return total_loss


def train_cross(loader, data_method, model_method):
    model.train()
    total_loss = 0
    model_aug(model_method)
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        x1, edge_index1, x2, edge_index2, pe = data_aug(data.x, data.batch, data.pe, data.edge_index, data_method)
        _, g1, g2 = model(data.x, x1, x2, edge_index1, edge_index2, pe, data.edge_index, data.edge_attr,
                          data.batch)
        _, g3, g4 = model_new(data.x, x1, x2, edge_index1, edge_index2, pe, data.edge_index, data.edge_attr,
                              data.batch)
        g1, g2, g3, g4 = [projector(g) for g in [g1, g2, g3, g4]]
        g3 = Variable(g3.detach().data, requires_grad=False)
        g4 = Variable(g4.detach().data, requires_grad=False)
        train_loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        train_loss += contrast_model(g1=g3, g2=g4, batch=data.batch)
        train_loss += contrast_model(g1=g1, g2=g3, batch=data.batch)
        train_loss += contrast_model(g1=g2, g2=g4, batch=data.batch)
        train_loss.backward()
        total_loss += train_loss.item()
        optimizer.step()
    return total_loss


@torch.no_grad()
def test():
    model.eval()
    x = []
    y = []
    for data in all_loader:
        data = data.to(device)
        optimizer.zero_grad()
        g, _, _ = model(data.x, data.x, data.x, data.edge_index, data.edge_index, data.pe, data.edge_index,
                        data.edge_attr, data.batch)
        x.append(g)
        y.append(data.y)

    X = torch.cat(x, dim=0)
    Y = torch.cat(y, dim=0)
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    acc, std = svc_classify(args.fold, X, Y, search=True)
    return acc, std


if __name__ == "__main__":
    run_name = f"{args.dataset}"
    best_loss = 1e9
    accs = []
    args.save_path = f"exps/{run_name}-{now}"
    os.makedirs(os.path.join(args.save_path, str(args.seed)), exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        # cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.aug == "data":
        train = train_data
        print("Start data aug pretrain")
    elif args.aug == "model":
        train = train_model
        print("Start model aug pretrain")
    elif args.aug == "cross":
        train = train_cross
        print("Start cross aug pretrain")
    pretrained_bar = tqdm(total=args.epochs, position=0)
    for epoch in range(1, args.epochs + 1):
        loss = train(all_loader, args.data_method, args.model_method)
        # val_acc = test()
        # state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        if loss < best_loss:
            best_loss = loss
            state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
            torch.save(state_dict, os.path.join(args.save_path, str(args.seed), "best_model.pt"))

        pretrained_bar.set_postfix({'loss': loss})
        pretrained_bar.set_description(f'Epoch {epoch}')
        pretrained_bar.update(1)
    pretrained_bar.close()
    state_dict = torch.load(os.path.join(args.save_path, str(args.seed), "best_model.pt"))
    model.load_state_dict(state_dict["model"])

    # unsupervised test
    acc, std = test()
    print('Acc: {:.4f} ± {:.4f}'.
          format(acc, std))
    save_accs(args, acc, std)
