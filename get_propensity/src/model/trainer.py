import json
import os

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


# Propensity of a user-item pair observed in an auxiliary behavior is shifted up
# by this offset, and of an unobserved pair down, before clipping to [0, 1].
OBS_OFFSET = 0.7


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
        self.out_dir = os.path.join(args.out_dir, args.dataset)
        os.makedirs(self.out_dir, exist_ok=True)
        self.aux_observed_mask = self._build_aux_observed_mask()

    def _build_aux_observed_mask(self):
        """Mask of user-item pairs observed in any auxiliary behavior."""
        data_path = os.path.join(self.args.data_dir, self.args.dataset)
        with open(os.path.join(data_path, 'statistics.json')) as f:
            bsg_types = json.load(f)['bsg_types']

        mask = torch.zeros(self.model.n_users + 1, self.model.n_items + 1, dtype=torch.bool)
        for behavior in bsg_types:
            if behavior in ('glo', 'buy'):
                continue
            edge_path = os.path.join(data_path, f'{behavior}.txt')
            if not os.path.exists(edge_path):
                continue
            edges = np.loadtxt(edge_path, dtype=int)
            mask[edges[:, 0], edges[:, 1]] = True
        return mask

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

        save_path = self.save_propensity_scores()
        logger.info(f'Saved propensity scores to {save_path}')

    def save_propensity_scores(self):
        self.model.eval()

        with torch.no_grad():
            aux_ob_emb = self.model.forward()
            user_emb, item_emb = torch.split(
                aux_ob_emb, [self.model.n_users + 1, self.model.n_items + 1], dim=0
            )
            scores = torch.sigmoid(torch.matmul(user_emb, item_emb.transpose(0, 1))).cpu()

        mask = self.aux_observed_mask
        H = min(scores.shape[0], mask.shape[0])
        W = min(scores.shape[1], mask.shape[1])
        scores_final = torch.clamp(
            torch.where(mask[:H, :W], scores[:H, :W] + OBS_OFFSET, scores[:H, :W] - OBS_OFFSET),
            min=0.0, max=1.0
        )

        propensity = scores_final.numpy().astype(np.float16)
        save_path = os.path.join(self.out_dir, 'propensity_scores.npy')
        np.save(save_path, propensity)
        return save_path
