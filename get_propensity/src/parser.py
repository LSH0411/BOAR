import argparse

from config import apply_config, config_pre_parser


def parse_args():
    pre_parser = config_pre_parser(default_dataset='jdata')
    known_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description='Propensity network settings', parents=[pre_parser]
    )
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing the data')
    parser.add_argument('--out_dir', type=str, default='./propensity_scores',
                        help='Directory where propensity score files are written')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for target data')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers')
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--neg_sample', type=int, default=1, help='Negative samples per positive interaction')

    apply_config(parser, known_args, 'propensity')
    return parser.parse_args()
