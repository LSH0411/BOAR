import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from .graph_conv import GraphConvLayer


class GRL(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        return -torch.log(self.gamma + torch.sigmoid(p_score - n_score)).mean()


class LearnableEdgeSelector(nn.Module):
    def __init__(self, emb_dim, gumbel_temp=0.2, sigma=1.0):
        super(LearnableEdgeSelector, self).__init__()
        self.gumbel_temp = gumbel_temp
        self.sigma = sigma
        self.mlp = nn.Linear(emb_dim * 2, 2)

    def forward(self, u_emb_all, i_emb_all, edge_index):
        u_emb = u_emb_all[edge_index[0]]
        i_emb = i_emb_all[edge_index[1]]
        u_norm = F.normalize(u_emb, p=2, dim=1)
        i_norm = F.normalize(i_emb, p=2, dim=1)
        cos_sim = torch.sum(u_norm * i_norm, dim=1)
        dist_sq = (u_emb - i_emb).pow(2).sum(dim=-1)
        rbf = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        sim_weight = 0.5 * (1 + cos_sim) * rbf
        logits = self.mlp(torch.cat([u_emb, i_emb], dim=1))
        gumbel_out = F.gumbel_softmax(logits, tau=self.gumbel_temp, hard=True)
        selection_mask = gumbel_out[:, 0]
        edge_weight = selection_mask * sim_weight.clamp(min=0)
        return edge_weight, selection_mask


class DensificationModule(nn.Module):
    def __init__(self, data, emb_dim, gnn_layers, dataset, args):
        super(DensificationModule, self).__init__()
        self.dataset = dataset
        self.device = args.device
        self.data_dir = args.data_dir
        self.edge_dict = data['edge_dict']
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.emb_dim = emb_dim
        self.gnn_layers = gnn_layers
        self.bsg_types = data['bsg_types']
        self.total_behaviors = ['glo'] + self.bsg_types
        self.target_behavior = 'buy'
        self.sigma = args.sigma
        self.dropout = args.dropout_dense

        if dataset == 'taobao':
            self.aux_behaviors = ['cart', 'view']
        else:
            self.aux_behaviors = ['cart', 'collect', 'view']
        self.num_domains = 2

        self.bpr_loss = BPRLoss()

        self.grl_lambda = args.grl_lambda
        self.sample_size = args.sample_size
        self.adv_w = args.adv_w
        self.lsh_top_k = args.lsh_top_k
        self.cl_temp = args.dense_cl_temp
        self.cl_w = args.dense_cl_w
        self.gumbel_temp = args.dense_gumbel_temp
        self.snips = args.snips
        self.scale = args.scale
        self.cl_generator = torch.Generator(device=self.device)
        self.cl_generator.manual_seed(args.seed + 9999)

        self.lsh_generator = torch.Generator(device=self.device)
        self.lsh_base_seed = args.seed
        self.lsh_call_count = 0
        self.lsh_rot_dim = emb_dim // 2

        self.edge_selector = LearnableEdgeSelector(emb_dim, gumbel_temp=self.gumbel_temp, sigma=self.sigma)

        self.user_embedding = nn.Embedding(self.n_users + 1, emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, emb_dim, padding_idx=0)
        
        self.convs = nn.ModuleDict()
        for behavior_type in self.total_behaviors:
            self.convs[behavior_type] = nn.ModuleList([
                GraphConvLayer(emb_dim, emb_dim, 'gcn', sigma=self.sigma, dropout=self.dropout)
                for _ in range(self.gnn_layers)
            ])

        self.dense_convs = nn.ModuleList([
            GraphConvLayer(emb_dim, emb_dim, 'dense', sigma=self.sigma, dropout=self.dropout)
            for _ in range(1)
        ])
        self.domain_discriminator = nn.Linear(emb_dim, 1)  
        self.domain_loss_fn = nn.BCEWithLogitsLoss()  
        self._build_observed_mask()
        self._build_item_popularity_labels()
        self._load_propensity_scores(self._resolve_propensity_path(args), args.device)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _build_observed_mask(self):
        glo_adj = self.edge_dict['glo']
        num_edges = glo_adj.size(1)
        half_num = num_edges // 2
        observed_users = glo_adj[0, :half_num]
        observed_items = glo_adj[1, :half_num] - (self.n_users + 1)

        self.observed_mask = torch.zeros(
            (self.n_users + 1, self.n_items + 1),
            dtype=torch.bool, device=self.device,
        )
        self.observed_mask[observed_users, observed_items] = True
        
    def _build_item_popularity_labels(self):
        glo_adj = self.edge_dict['aux_ob'].cpu()
        half_num = glo_adj.size(1) // 2
        glo_items = glo_adj[1, :half_num] - (self.n_users + 1)

        item_counts = torch.zeros(self.n_items + 1, dtype=torch.float)
        item_counts.scatter_add_(0, glo_items, torch.ones(glo_items.size(0), dtype=torch.float))

        median_count = item_counts[1:].median()  
        self.item_pop_labels = (item_counts >= median_count).long() 
        self.item_pop_labels[0] = -1  

        self._pop_labels_on_gpu = False

    @staticmethod
    def _resolve_propensity_path(args):
        if args.propensity_path is not None:
            return args.propensity_path
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(
            repo_root, 'get_propensity', 'propensity_scores', args.dataset, 'propensity_scores.npy'
        )

    def _load_propensity_scores(self, propensity_path, device):
        if not os.path.exists(propensity_path):
            raise FileNotFoundError(f"Propensity scores not found at {propensity_path}")
        propensity_matrix = np.load(propensity_path)
        propensity_matrix = np.clip(propensity_matrix, 1e-5, 1.0)
        self.propensity_scores = torch.from_numpy(propensity_matrix).float().pin_memory()

    
    def _sample_adversarial_pairs(self, batch_size_per_class):
        device = self.user_embedding.weight.device

        if not self._pop_labels_on_gpu:
            self.item_pop_labels = self.item_pop_labels.to(device)
            self._pop_labels_on_gpu = True

        valid_labels = self.item_pop_labels[1:]
        popular_items = torch.where(valid_labels == 1)[0] + 1
        non_popular_items = torch.where(valid_labels == 0)[0] + 1

        sampled_items_list, sampled_labels_list = [], []

        for label, pool in [(1, popular_items), (0, non_popular_items)]:
            if pool.size(0) > 0:
                num_samples = min(batch_size_per_class, pool.size(0))
                indices = torch.randint(0, pool.size(0), (num_samples,), device=device)
                sampled_items_list.append(pool[indices])
                sampled_labels_list.append(
                    torch.full((num_samples,), label, dtype=torch.long, device=device)
                )

        items = torch.cat(sampled_items_list, dim=0)
        labels = torch.cat(sampled_labels_list, dim=0)
        return items, labels

    def _propagate(self, x, edge_index, behavior_type):
        result = [x]
        for i, conv in enumerate(self.convs[behavior_type]):
            x = conv(x, edge_index)
            x = F.normalize(x, dim=-1)
            result.append(x / (i + 1))
        result = torch.stack(result, dim=0)
        return result.sum(dim=0)

    def _propagate_dense(self, x, edge_index, edge_weight):
        result = [x]
        for i, conv in enumerate(self.dense_convs):
            x = conv(x, edge_index, edge_weight)
            x = F.normalize(x, dim=-1)
            result.append(x / (i + 1))
        result = torch.stack(result, dim=0)
        return result.sum(dim=0)

    def _retrieve_lsh_candidates(self, user_emb, item_emb):
        u_norm = F.normalize(user_emb, p=2, dim=1)
        i_norm = F.normalize(item_emb, p=2, dim=1)
        valid_u = u_norm[1:]
        valid_i = i_norm[1:]

        self.lsh_generator.manual_seed(self.lsh_base_seed + self.lsh_call_count)
        random_rot = torch.randn(
            self.emb_dim, self.lsh_rot_dim,
            device=self.device, generator=self.lsh_generator,
        )
        self.lsh_call_count += 1
        
        def get_bucket_indices(x):
            rotated = torch.matmul(x, random_rot)
            rotated = torch.cat([rotated, -rotated], dim=-1)
            return torch.argmax(rotated, dim=-1)

        u_buckets = get_bucket_indices(valid_u)
        i_buckets = get_bucket_indices(valid_i)

        cand_u_list, cand_i_list, cand_v_list = [], [], []
        unique_buckets = torch.unique(u_buckets)

        for bucket_id in unique_buckets:
            curr_u_indices = torch.where(u_buckets == bucket_id)[0]
            curr_i_indices = torch.where(i_buckets == bucket_id)[0]

            if len(curr_i_indices) == 0:
                continue

            b_u = valid_u[curr_u_indices]
            b_i = valid_i[curr_i_indices]
            sim_matrix = torch.mm(b_u, b_i.t())
            sim_matrix = (sim_matrix + 1) / 2

            global_u_ids = curr_u_indices + 1
            global_i_ids = curr_i_indices + 1
            observed_in_bucket = self.observed_mask[global_u_ids[:, None], global_i_ids[None, :]]
            sim_matrix = sim_matrix.masked_fill(observed_in_bucket, float('-inf'))

            curr_k = min(self.lsh_top_k, len(curr_i_indices))
            if curr_k == 0:
                continue

            vals, local_inds = torch.topk(sim_matrix, k=curr_k, dim=1)
            valid_mask = vals > float('-inf')

            if valid_mask.sum() == 0:
                continue

            vals_flat = vals[valid_mask]
            user_repeat_idx = torch.arange(len(curr_u_indices), device=self.device)
            user_repeat_idx = user_repeat_idx.unsqueeze(1).expand_as(local_inds)
            user_repeat_idx_flat = user_repeat_idx[valid_mask]
            local_inds_flat = local_inds[valid_mask]

            global_u_ids_final = curr_u_indices[user_repeat_idx_flat] + 1
            global_i_ids_final = curr_i_indices[local_inds_flat] + 1

            cand_u_list.append(global_u_ids_final)
            cand_i_list.append(global_i_ids_final)
            cand_v_list.append(vals_flat)

        if len(cand_u_list) > 0:
            all_users = torch.cat(cand_u_list, dim=0)
            all_items = torch.cat(cand_i_list, dim=0)
            global_item_ids_gnn = all_items + (self.n_users + 1)
            cand_edge_index = torch.stack([all_users, global_item_ids_gnn], dim=0)
            return cand_edge_index, torch.cat(cand_v_list, dim=0)
        else:
            empty_edges = torch.zeros(2, 0, device=self.device, dtype=torch.long)
            empty_vals = torch.tensor([], device=self.device)
            return empty_edges, empty_vals

    def _build_augmented_graph(self, emb_glo):
        orig_edge_index = self.edge_dict[self.target_behavior]
        half_idx = orig_edge_index.size(1) // 2
        orig_ui_index = orig_edge_index[:, :half_idx]
        orig_weight = torch.ones(orig_ui_index.size(1), device=self.device)

        user_emb_all, item_emb_all = torch.split(
            emb_glo, [self.n_users + 1, self.n_items + 1]
        )
        cand_edge_index, _ = self._retrieve_lsh_candidates(user_emb_all, item_emb_all)

        if cand_edge_index.size(1) > 0:
            cand_u_local = cand_edge_index[0]
            cand_i_local = cand_edge_index[1] - (self.n_users + 1)
            cand_local_index = torch.stack([cand_u_local, cand_i_local], dim=0)

            cand_weight, _ = self.edge_selector(user_emb_all, item_emb_all, cand_local_index)

            mask = cand_weight > 1e-9
            final_cand_edge = cand_edge_index[:, mask]
            final_cand_weight = cand_weight[mask]
        else:
            final_cand_edge = torch.zeros(2, 0, device=self.device, dtype=torch.long)
            final_cand_weight = torch.tensor([], device=self.device)

        merged_ui_index = torch.cat([orig_ui_index, final_cand_edge], dim=1)
        merged_weight = torch.cat([orig_weight, final_cand_weight], dim=0)

        merged_iu_index = torch.stack([merged_ui_index[1], merged_ui_index[0]], dim=0)
        full_edge_index = torch.cat([merged_ui_index, merged_iu_index], dim=1)
        full_edge_weight = torch.cat([merged_weight, merged_weight], dim=0)

        return full_edge_index, full_edge_weight

    def _adversarial_loss(self, item_emb, items, labels, grl_lambda=0.1):
        item_emb_grl = GRL.apply(item_emb, grl_lambda)
        logits = self.domain_discriminator(item_emb_grl[items]).squeeze(-1)
        loss = self.domain_loss_fn(logits, labels.float())

        with torch.no_grad():
            pred = (logits > 0).long()
            acc_dict = {'overall': (pred == labels).float().mean().item()}

            pop_mask = labels == 1
            if pop_mask.sum() > 0:
                acc_dict['popular'] = (pred[pop_mask] == 1).float().mean().item()

            nonpop_mask = labels == 0
            if nonpop_mask.sum() > 0:
                acc_dict['non_popular'] = (pred[nonpop_mask] == 0).float().mean().item()

        return loss, acc_dict

    def forward(self):
        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        emb_glo = self._propagate(init_emb, self.edge_dict['glo'], 'glo')
        aug_edge_index, aug_edge_weight = self._build_augmented_graph(emb_glo)
        emb_dense = self._propagate_dense(emb_glo, aug_edge_index, edge_weight=aug_edge_weight)
        target_edge = self.edge_dict[self.target_behavior]
        emb_target = self._propagate(init_emb, target_edge, self.target_behavior)
        return emb_glo, emb_target, emb_dense

    def forward_pred(self):
        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        target_edge = self.edge_dict[self.target_behavior]
        emb_target = self._propagate(init_emb, target_edge, self.target_behavior)
        return emb_target

    def loss(self, user_indices, pos_indices, neg_indices):
        emb_glo, emb_target, emb_dense = self.forward()
        u_glo, i_glo = torch.split(emb_glo, [self.n_users + 1, self.n_items + 1], dim=0)
        u_target, i_target = torch.split(emb_target, [self.n_users + 1, self.n_items + 1], dim=0)
        u_dense, i_dense = torch.split(emb_dense, [self.n_users + 1, self.n_items + 1], dim=0)
        
        items, labels = self._sample_adversarial_pairs(self.sample_size)
        adv_loss, _ = self._adversarial_loss(
            i_glo, items, labels, grl_lambda=self.grl_lambda
        )
        
        epsilon = 1e-8
        pos_propensity = self.propensity_scores[user_indices.cpu(), pos_indices.cpu()].to(self.device, non_blocking=True)
        ips_weight = (1.0 / (pos_propensity + epsilon)) ** self.scale
        ips_weight = torch.clamp(ips_weight, max=self.snips)
        normalized_ips_weight = ips_weight / (torch.sum(ips_weight) + epsilon)

        p_score = torch.einsum('ij,ij->i', u_target[user_indices], i_target[pos_indices])
        n_score = torch.einsum('ij,ij->i', u_target[user_indices], i_target[neg_indices])
        element_wise_bpr = -torch.log(torch.sigmoid(p_score - n_score) + epsilon)
        bpr_loss = torch.sum(element_wise_bpr * normalized_ips_weight.detach())

        cl_u = self._contrastive_loss(u_target, u_dense, self.cl_temp)
        cl_i = self._contrastive_loss(i_target, i_dense, self.cl_temp)
        cl_loss = cl_u + cl_i

        total_loss = bpr_loss + self.adv_w * adv_loss + self.cl_w * cl_loss
        return total_loss, bpr_loss

    @torch.no_grad()
    def precompute_embeddings(self):
        emb_target = self.forward_pred()
        self._cached_user_emb, self._cached_item_emb = torch.split(
            emb_target, [self.n_users + 1, self.n_items + 1]
        )
    @torch.no_grad()
    def predict(self, users):
        user_emb = self._cached_user_emb[users.long()]
        return torch.mm(user_emb, self._cached_item_emb.t())

    def _contrastive_loss(self, pos, aug, temp):
        sampled_indices = torch.randperm(pos.shape[0], generator=self.cl_generator, device=self.device)[:1024]
        pos = F.normalize(pos[sampled_indices], p=2, dim=1)
        aug = F.normalize(aug[sampled_indices], p=2, dim=1)

        pos_score = torch.sum(pos * aug, dim=1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(pos, aug.permute(1, 0))
        ttl_score = torch.sum(torch.exp(ttl_score / temp), axis=1)

        return -torch.mean(torch.log(pos_score / ttl_score))
