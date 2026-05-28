from data import load_data
from model import RefinementModule, DensificationModule, BOARTrainer
from parser import parse_args
from utils import set_seed
from loguru import logger


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    data = load_data(args.data_dir, args.dataset, args.device, args.batch_size, seed=args.seed)

    refinement_module = RefinementModule(
        data, args.emb_dim, args.gnn_layers, args.dataset, args
    ).to(args.device)
    densification_module = DensificationModule(
        data, args.emb_dim, args.gnn_layers, args.dataset, args
    ).to(args.device)

    trainer = BOARTrainer(refinement_module, densification_module, data, args)

    logger.info('Start training BOAR')
    trainer.train_model()


if __name__ == '__main__':
    args = parse_args()
    main(args)
