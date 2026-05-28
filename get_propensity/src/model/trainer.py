import json
import torch
import numpy as np
import os
from tqdm import tqdm
from loguru import logger

class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        train_loader = self.data['aux_ob_train_loader']

        for user_indices, pos_indices, neg_indices in tqdm(
            train_loader, desc=f'Epoch {epoch+1}', leave=False
        ):
            user_indices = user_indices.to(self.args.device)
            pos_indices = pos_indices.to(self.args.device)
            neg_indices = neg_indices.to(self.args.device)

            loss = self.model.loss(user_indices, pos_indices, neg_indices)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(self):
        for epoch in range(self.args.num_epochs):
            avg_loss = self.train_epoch(epoch)
            logger.info(f'Epoch {epoch+1}/{self.args.num_epochs}  loss: {avg_loss:.4f}')

    def save_propensity_scores(self, save_path):
        self.model.eval()

        with torch.no_grad():
            aux_ob_emb = self.model.forward()
            user_emb, item_emb = torch.split(
                aux_ob_emb, [self.model.n_users + 1, self.model.n_items + 1], dim=0
            )
            scores = torch.sigmoid(torch.matmul(user_emb, item_emb.transpose(0, 1))).cpu()

        data_path = os.path.join(self.args.data_dir, self.args.dataset)
        with open(os.path.join(data_path, 'statistics.json')) as f:
            bsg_types = json.load(f)['bsg_types']

        n_u, n_i = self.model.n_users, self.model.n_items
        mask = torch.zeros(n_u + 1, n_i + 1, dtype=torch.bool)
        for beh in bsg_types:
            if beh in ('glo', 'buy'):
                continue
            epath = os.path.join(data_path, f'{beh}.txt')
            if not os.path.exists(epath):
                continue
            edges = np.loadtxt(epath, dtype=int)
            mask[edges[:, 0], edges[:, 1]] = True

        H = min(scores.shape[0], mask.shape[0])
        W = min(scores.shape[1], mask.shape[1])
        scores_final = torch.clamp(
            torch.where(mask[:H, :W], scores[:H, :W] + 0.7, scores[:H, :W] - 0.7),
            min=0.0, max=1.0
        )

        base_dir = os.path.dirname(os.path.dirname(save_path))
        dataset_dir = os.path.basename(os.path.dirname(save_path))
        path = os.path.join(base_dir, dataset_dir, os.path.basename(save_path))
        os.makedirs(os.path.dirname(path), exist_ok=True)

        propensity = scores_final.numpy().astype(np.float16)
        np.save(path, propensity)
        return propensity