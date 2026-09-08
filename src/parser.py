import argparse

from config import apply_config, config_pre_parser

def parse_args():
    pre_parser = config_pre_parser(default_dataset='jdata')
    known_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description='BOAR: Beyond Observed Auxiliary Relations', parents=[pre_parser]
    )
    # Data / experiment
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Directory of model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load model checkpoint')
    parser.add_argument('--log_dir', type=str, default='./log', help='Directory for training logs')
    parser.add_argument('--propensity_path', type=str, default=None,
                        help='Propensity score file (default: get_propensity/propensity_scores/<dataset>/propensity_scores.npy)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--topk', type=int, default=10, help='Top-k for evaluation')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')

    # Backbone
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--emb_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--sigma', type=float, default=20.0, help='RBF bandwidth for edge reweighting')

    # Densification module (f0)
    parser.add_argument('--dense_cl_w', type=float, default=0.2, help='Weight for densification contrastive loss (lambda_dense)')
    parser.add_argument('--dense_cl_temp', type=float, default=0.1, help='Temperature for densification contrastive loss')
    parser.add_argument('--dense_gumbel_temp', type=float, default=0.2, help='Gumbel-softmax temperature of the edge selector')
    parser.add_argument('--lsh_top_k', type=int, default=30, help='Top-k candidates per LSH bucket')
    parser.add_argument('--dropout_dense', type=float, default=0.4, help='Edge dropout in the densification module')
    parser.add_argument('--grl_lambda', type=float, default=0.1, help='Gradient reversal strength of the popularity discriminator')
    parser.add_argument('--adv_w', type=float, default=0.01, help='Weight for the adversarial popularity loss')
    parser.add_argument('--sample_size', type=int, default=1024, help='Items sampled per popularity class for the adversarial loss')
    parser.add_argument('--snips', type=float, default=5.0, help='Clipping threshold of the self-normalized IPS weights')
    parser.add_argument('--scale', type=float, default=0.5, help='Exponent applied to the inverse propensity weights')

    # Refinement module (f1)
    parser.add_argument('--refine_cl_w', type=float, default=0.2, help='Weight for refinement contrastive loss (lambda_refine)')
    parser.add_argument('--refine_cl_temp', type=float, default=1.5, help='Temperature for refinement contrastive loss')
    parser.add_argument('--prune_temp', type=float, default=0.2, help='Gumbel-softmax temperature of the edge pruner')
    parser.add_argument('--prune_init', type=float, default=2.0, help='Keep/drop bias initialization of the edge pruner')
    parser.add_argument('--dropout_refine', type=float, default=0.1, help='Edge dropout in the refinement module')

    parser.set_defaults(**DATASET_DEFAULTS.get(known_args.dataset, {}))
    apply_config(parser, known_args, 'boar')
    return parser.parse_args()

DATASET_DEFAULTS = {
    'taobao': {'snips': 5.0, 'scale': 0.5, 'prune_init': 2.0, 'dropout_refine': 0.4},
    'jdata': {'snips': 5.0, 'scale': 0.5, 'prune_init': 2.0, 'dropout_refine': 0.1},
    'tmall': {'snips': 100.0, 'scale': 1.0, 'prune_init': 0.5, 'dropout_refine': 0.1},
}