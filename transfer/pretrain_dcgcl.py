import argparse
import os

from torch.nn.utils import prune
from torch_geometric.nn import global_mean_pool

from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
import torch_geometric.transforms as T

from copy import deepcopy

from posen import AddRandomWalkPE


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x * h, dim=1)


def model_aug(model, aug, ratio):
    if aug == "Gaussian":
        std = 0.01
        mean = 0.0
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.normal(mean=mean, std=std, size=param.size())
                param.data.add_(noise)
    elif aug == "DropWeight":
        parameters_to_prune = []
        for name, module in model.named_modules():
            if not list(module.children()):
                for param_name, param in module.named_parameters():
                    if param_name == "weight":
                        parameters_to_prune.append((module, param_name))
        for module, param_name in parameters_to_prune:
            prune.l1_unstructured(module, param_name, amount=ratio)
            prune.remove(module, param_name)
    elif aug == "DropHead":
        for conv in model.gnn.gnns:
            conv.conv.attn_drophead = ratio

    return model


class CGT(nn.Module):

    def __init__(self, gnn):
        super(CGT, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, pe, edge_index, edge_attr, batch):
        x = self.gnn(x, pe, edge_index, edge_attr, batch)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def train(args, model, device, dataset, optimizer):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1.x, batch1.pe, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.pe, batch2.edge_index, batch2.edge_attr, batch2.batch)
        model_new = deepcopy(model).to(device)
        model_new = model_aug(model_new, args.model_aug, args.model_ratio)
        x3 = model_new.forward_cl(batch1.x, batch1.pe, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x4 = model_new.forward_cl(batch2.x, batch2.pe, batch2.edge_index, batch2.edge_attr, batch2.batch)
        # 4 losses
        loss = model.loss_cl(x1, x2)
        loss += model.loss_cl(x1, x3)
        loss += model.loss_cl(x2, x4)
        loss += model.loss_cl(x3, x4)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum / (step + 1), train_loss_accum / (step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 1).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type=str, default='',
                        help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="transformer")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default='none')
    parser.add_argument('--aug_ratio1', type=float, default=0.2)
    parser.add_argument('--aug2', type=str, default='PEMask')
    parser.add_argument('--aug_ratio2', type=float, default=0.2)
    parser.add_argument('--model_aug', type=str, default='DropHead')
    parser.add_argument('--model_ratio', type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    transform = AddRandomWalkPE(20)
    # set up dataset
    dataset = MoleculeDataset_aug("./dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)

    pe_list = []
    os.makedirs("./pe", exist_ok=True)
    if not os.path.exists("./pe/" + args.dataset + "_pe.pt"):
        for data in tqdm(dataset, desc="Preprocessing"):
            pe = transform(data.edge_index, data.num_nodes)
            pe_list.append(pe)
        dataset.pe = pe_list
        torch.save(pe_list, "./pe/" + args.dataset + "_pe.pt")
    else:
        dataset.pe = torch.load("./pe/" + args.dataset + "_pe.pt")
    # set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    model = CGT(gnn)

    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        _, train_loss = train(args, model, device, dataset, optimizer)
        print(train_loss)

        if epoch % 10 == 0:
            os.makedirs("./pretrained", exist_ok=True)
            torch.save(gnn.state_dict(), "./pretrained/CGT" + str(epoch) + ".pth")


if __name__ == "__main__":
    main()
