import csv
import os

import configargparse


def gene_arg():
    parser = configargparse.ArgumentParser(allow_abbrev=False,
                                           description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--configs', required=False, is_config_file=True)
    parser.add_argument('--wandb_run_idx', type=str, default=None)

    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default="COLLAB",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--aug', type=str, default='cross',
                        help='augment method to use [data|model|cross]')

    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='maximum sequence length to predict (default: None)')

    group = parser.add_argument_group('model')
    group.add_argument('--model_type', type=str, default='gnn', help='gnn|pna|gnn-transformer')
    group.add_argument('--graph_pooling', type=str, default='mean')
    group = parser.add_argument_group('gnn')
    group.add_argument('--gnn_type', type=str, default='gcn')
    group.add_argument('--gnn_virtual_node', action='store_true')
    group.add_argument('--gnn_dropout', type=float, default=0)
    group.add_argument('--gnn_num_layer', type=int, default=5,
                       help='number of GNN message passing layers (default: 5)')
    group.add_argument('--gnn_emb_dim', type=int, default=300,
                       help='dimensionality of hidden units in GNNs (default: 300)')
    group.add_argument('--gnn_JK', type=str, default='last')
    group.add_argument('--channels', type=int, default=64)
    group.add_argument('--data_method', type=str, default='MAE')
    group.add_argument('--model_method', type=str, default='Gaussian')
    group.add_argument('--aug_ratio', type=float, default=0.2)
    group.add_argument('--gnn_residual', action='store_true', default=False)
    group.add_argument('--num_layers', type=int, default=5,
                       help='number of GNN message passing layers (default: 5)')
    group.add_argument('--nhead', type=int, default=5,
                       help='number of GNN message passing layers (default: 5)')

    group = parser.add_argument_group('training')
    group.add_argument('--devices', type=int, default=0,
                       help='which gpu to use if any (default: 0)')
    group.add_argument('--batch_size', type=int, default=128,
                       help='input batch size for training (default: 128)')
    group.add_argument('--eval_batch_size', type=int, default=128,
                       help='input batch size for training (default: train batch size)')
    group.add_argument('--epochs', type=int, default=30,
                       help='number of epochs to train (default: 30)')
    parser.add_argument('--semi_split', type=int, default=10, help='10-fold or 100-fold')
    group.add_argument('--semi_epochs', type=int, default=100)
    group.add_argument('--fold', type=int, default=10)
    group.add_argument('--num_workers', type=int, default=0,
                       help='number of workers (default: 0)')
    group.add_argument('--scheduler', type=str, default=None)
    group.add_argument('--pct_start', type=float, default=0.3)
    group.add_argument('--weight_decay', type=float, default=0.0)
    group.add_argument('--grad_clip', type=float, default=None)
    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--max_lr', type=float, default=0.001)
    group.add_argument('--runs', type=int, default=10)
    group.add_argument('--test-freq', type=int, default=1)
    group.add_argument('--start-eval', type=int, default=15)
    group.add_argument('--resume', type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    group.add_argument('--run', type=int, default=0)

    # fmt: on

    args, _ = parser.parse_known_args()

    return args

def save_accs(args, acc, std):
    save_path = os.path.join(f"./checkpoint/{args.dataset}/{args.aug}/{args.num_layers}")
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"{args.num_layers}.csv")
    header = ["data_method", "model_method", "acc", "std"]

    if not os.path.isfile(path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

    with open(path, 'a', newline='') as csvfile:
        line = "{}, {}, {:.4f}, {:.4f}".format(args.data_method, args.model_method, acc, std)
        csvfile.write(line + "\n")