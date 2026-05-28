import os
import torch
from data import load_data
from model import LGCN_G, Trainer
from parser import parse_args
from utils import set_seed
from loguru import logger

def main(args):
    if args.seed is not None:
        set_seed(args.seed)
    
    # Load data
    data = load_data(args.data_dir, args.dataset, args.device, args.batch_size, args.neg_sample)
    
    # Build model
    model = LGCN_G(data, args.emb_dim, args.gnn_layers, args.dataset, args).to(args.device)
    trainer = Trainer(model, data, args)
    
    # Train model
    logger.info("Start training the model")
    trainer.train()

    # Save propensity scores
    save_path = f'./propensity_scores/{args.dataset}/propensity_scores.npy'
    trainer.save_propensity_scores(save_path)
if __name__ == '__main__':
    args = parse_args()
    
    main(args)