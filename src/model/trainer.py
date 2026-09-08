import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from .metrics import ndcg, hit


class BOARTrainer:
    def __init__(self, refinement_module, densification_module, data, args):
        self.refinement_module = refinement_module
        self.densification_module = densification_module
        self.data = data
        self.args = args
        self.topk = args.topk
        self.refine_optimizer = torch.optim.Adam(
            self.refinement_module.parameters(),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        self.dense_optimizer = torch.optim.Adam(
            self.densification_module.parameters(),
            lr=args.lr, weight_decay=args.weight_decay,
        )

        self._build_auxiliary_masks()

        # Logging
        log_dir = os.path.join(args.log_dir, args.dataset)
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(log_dir, f'boar_training_{now}.log')

        self.logger = logging.getLogger(f'boar_{now}')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.propensity_scores = self.densification_module.propensity_scores

        self.logger.info('===== BOAR Experiment Arguments =====')
        for key, value in vars(args).items():
            self.logger.info(f'{key}: {value}')

    def _build_auxiliary_masks(self):
        device = self.args.device
        n_users = self.data['n_users']
        n_items = self.data['n_items']

        self.observed_mask = torch.zeros(n_users + 1, n_items + 1, dtype=torch.bool, device=device)

        observed_interactions = {}
        for behavior in self.data['edge_dict']:
            if behavior in ('glo', 'buy'):
                continue
            adj = self.data['edge_dict'][behavior].cpu()
            half_num = adj.size(1) // 2
            users = adj[0, :half_num].numpy()
            items = (adj[1, :half_num] - (n_users + 1)).numpy()
            for u, i in zip(users, items):
                if u not in observed_interactions:
                    observed_interactions[u] = set()
                observed_interactions[u].add(i)

        for user, items in observed_interactions.items():
            for item in items:
                self.observed_mask[user, item] = True

        self.unobserved_mask = ~self.observed_mask

    def _apply_environment_masks(self, scores_observed, scores_unobserved, user_indices):
        obs_mask_batch = self.observed_mask[user_indices].to(dtype=scores_observed.dtype)
        unobs_mask_batch = self.unobserved_mask[user_indices].to(dtype=scores_unobserved.dtype)
        return scores_observed * obs_mask_batch + scores_unobserved * unobs_mask_batch

    # ========== Training ==========

    def _train_epoch_refinement(self):
        self.refinement_module.train()
        total_loss = 0
        loader = self.data['train_loader']

        for user_indices, pos_indices, neg_indices in tqdm(loader, desc='[f1] Refinement', leave=False):
            user_indices = user_indices.to(self.args.device)
            pos_indices = pos_indices.to(self.args.device)
            neg_indices = neg_indices.to(self.args.device)

            loss = self.refinement_module.loss(user_indices, pos_indices, neg_indices)
            self.refine_optimizer.zero_grad()
            loss.backward()
            self.refine_optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _train_epoch_densification(self):
        self.densification_module.train()
        total_loss = 0
        total_adv_loss = 0
        total_disc_acc = None
        loader = self.data['train_loader']

        for user_indices, pos_indices, neg_indices in tqdm(loader, desc='[f0] Densification', leave=False):
            user_indices = user_indices.to(self.args.device)
            pos_indices = pos_indices.to(self.args.device)
            neg_indices = neg_indices.to(self.args.device)

            loss, bpr_loss = self.densification_module.loss(
                user_indices, pos_indices, neg_indices,
            )

            self.dense_optimizer.zero_grad()
            loss.backward()
            self.dense_optimizer.step()

            total_loss += loss.item()


        num_batches = len(loader)
        return total_loss / num_batches

    def train_model(self):
        num_epochs = self.args.num_epochs

        pbar = tqdm(range(num_epochs), desc='Epoch', unit='epoch')
        for epoch in pbar:
            refine_loss = self._train_epoch_refinement()
            dense_loss = self._train_epoch_densification()

            pbar.set_description(
                f'Epoch {epoch+1} | f1: {refine_loss:.4f} | f0: {dense_loss:.4f}'
            )

            log_msg = (
                f'Epoch {epoch+1} - '
                f'Refinement Loss: {refine_loss:.4f}, '
                f'Densification Loss: {dense_loss:.4f}, '
            )
            print(log_msg)
            self.logger.info(log_msg)

        # Evaluate after training
        metrics = self._evaluate_general()
        print(f'General    - HR@10: {metrics["hr@10"]:.4f}, NDCG@10: {metrics["ndcg@10"]:.4f}')
        self.logger.info(f'General    - HR@10: {metrics["hr@10"]:.4f}, NDCG@10: {metrics["ndcg@10"]:.4f}')

        metrics_obs = self._evaluate_observed()
        print(f'Observed   - HR@10: {metrics_obs["hr@10"]:.4f}, NDCG@10: {metrics_obs["ndcg@10"]:.4f}')
        self.logger.info(f'Observed   - HR@10: {metrics_obs["hr@10"]:.4f}, NDCG@10: {metrics_obs["ndcg@10"]:.4f}')

        metrics_unobs = self._evaluate_unobserved()
        print(f'Unobserved - HR@10: {metrics_unobs["hr@10"]:.4f}, NDCG@10: {metrics_unobs["ndcg@10"]:.4f}')
        self.logger.info(f'Unobserved - HR@10: {metrics_unobs["hr@10"]:.4f}, NDCG@10: {metrics_unobs["ndcg@10"]:.4f}')

    # ========== Evaluation ==========

    def _evaluate_general(self):
        device = self.args.device
        self.refinement_module.eval()
        self.densification_module.eval()
        self.refinement_module.precompute_embeddings()
        self.densification_module.precompute_embeddings()

        topk_list = []
        with torch.no_grad():
            for user_indices, _, _ in tqdm(self.data['test_loader'], desc='Eval (General)', leave=False):
                user_indices = user_indices.to(device)
                scores_observed = self.refinement_module.predict(user_indices)
                scores_unobserved = self.densification_module.predict(user_indices)
                propensity = self.propensity_scores[user_indices.cpu()].to(device, non_blocking=True)
                combined_scores = scores_observed * propensity + scores_unobserved * (1 - propensity)

                for user_idx in range(user_indices.size(0)):
                    user = user_indices[user_idx].item()
                    user_str = str(user)
                    user_score = combined_scores[user_idx]

                    train_items = self.data['train_gt'].get(user_str, [])
                    if len(train_items) > 0:
                        user_score[train_items] = -torch.inf

                    _, topk_indices = torch.topk(user_score, self.topk)
                    gt_items = np.array(self.data['test_gt'][user_str])
                    topk_items = topk_indices.cpu().numpy()
                    topk_list.append(np.isin(topk_items, gt_items))

        return self._compute_metrics(topk_list, self.data['test_gt_length'])

    def _evaluate_observed(self):
        device = self.args.device
        if 'test_observed_gt' not in self.data:
            return self._empty_metrics()

        self.refinement_module.eval()
        self.densification_module.eval()
        self.refinement_module.precompute_embeddings()
        self.densification_module.precompute_embeddings()
        topk_list = []
        test_observed_users = []

        with torch.no_grad():
            for user_indices, _, _ in tqdm(self.data['test_observed_loader'], desc='Eval (Observed)', leave=False):
                user_indices = user_indices.to(device)
                scores_observed = self.refinement_module.predict(user_indices)
                scores_unobserved = self.densification_module.predict(user_indices)
                propensity = self.propensity_scores[user_indices.cpu()].to(device, non_blocking=True)
                combined = scores_observed * propensity + scores_unobserved * (1 - propensity)
                unobs_mask_batch = self.unobserved_mask[user_indices]
                combined[unobs_mask_batch] = -torch.inf

                for user_idx in range(user_indices.size(0)):
                    user = user_indices[user_idx].item()
                    user_str = str(user)
                    if user_str not in self.data['test_observed_gt']:
                        continue

                    user_score = combined[user_idx]
                    train_items = self.data['train_gt'].get(user_str, [])
                    if len(train_items) > 0:
                        user_score[train_items] = -torch.inf

                    _, topk_indices = torch.topk(user_score, self.topk)
                    gt_items = np.array(self.data['test_observed_gt'][user_str])
                    topk_items = topk_indices.cpu().numpy()
                    topk_list.append(np.isin(topk_items, gt_items))
                    test_observed_users.append(user_str)

        if len(topk_list) == 0:
            return self._empty_metrics()

        gt_length = np.array([len(self.data['test_observed_gt'][u]) for u in test_observed_users])
        return self._compute_metrics(topk_list, gt_length)

    def _evaluate_unobserved(self):
        device = self.args.device
        if 'test_unobserved_gt' not in self.data:
            return self._empty_metrics()

        self.refinement_module.eval()
        self.densification_module.eval()
        self.refinement_module.precompute_embeddings()
        self.densification_module.precompute_embeddings()

        topk_list = []
        test_unobserved_users = []

        with torch.no_grad():
            for user_indices, _, _ in tqdm(self.data['test_unobserved_loader'], desc='Eval (Unobserved)', leave=False):
                user_indices = user_indices.to(device)
                scores_observed = self.refinement_module.predict(user_indices)
                scores_unobserved = self.densification_module.predict(user_indices)
                propensity = self.propensity_scores[user_indices.cpu()].to(device, non_blocking=True)
                combined = scores_observed * propensity + scores_unobserved * (1 - propensity)
                obs_mask_batch = self.observed_mask[user_indices]
                combined[obs_mask_batch] = -torch.inf

                for user_idx in range(user_indices.size(0)):
                    user = user_indices[user_idx].item()
                    user_str = str(user)
                    if user_str not in self.data['test_unobserved_gt']:
                        continue

                    user_score = combined[user_idx]
                    train_items = self.data['train_gt'].get(user_str, [])
                    if len(train_items) > 0:
                        user_score[train_items] = -torch.inf

                    _, topk_indices = torch.topk(user_score, self.topk)
                    gt_items = np.array(self.data['test_unobserved_gt'][user_str])
                    topk_items = topk_indices.cpu().numpy()
                    topk_list.append(np.isin(topk_items, gt_items))
                    test_unobserved_users.append(user_str)

        if len(topk_list) == 0:
            return self._empty_metrics()

        gt_length = np.array([len(self.data['test_unobserved_gt'][u]) for u in test_unobserved_users])
        return self._compute_metrics(topk_list, gt_length)

    def _compute_metrics(self, topk_list, gt_length):
        topk_array = np.vstack(topk_list)
        hr = hit(topk_array, gt_length).mean(axis=0)[self.topk - 1]
        ndcg_val = ndcg(topk_array, gt_length).mean(axis=0)[self.topk - 1]
        return {'hr@10': hr, 'ndcg@10': ndcg_val}

    def _empty_metrics(self):
        return {'hr@10': 0.0, 'ndcg@10': 0.0}