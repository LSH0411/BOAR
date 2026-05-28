import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_conv import GraphConvLayer
from .target_attn import TargetAttention


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss."""

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        return -torch.log(self.gamma + torch.sigmoid(p_score - n_score)).mean()
    
class GumbelEdgePruner(nn.Module):
    """Binary edge pruner via Gumbel-Softmax with feature refinement and RBF reweighting."""

    def __init__(self, emb_dim, gumbel_temp=0.2, sigma=20.0, init=2.0):
        super(GumbelEdgePruner, self).__init__()
        self.gumbel_temp = gumbel_temp
        self.sigma = sigma
        self.gate = nn.Linear(2 * emb_dim, emb_dim)
        self.mlp = nn.Linear(emb_dim*2, 2)
        with torch.no_grad():
            self.mlp.bias.data = torch.tensor([init, -init])  # keep >> drop

    def forward(self, emb_glo, emb_tar, edge_index):
        # (1) Feature refinement: gate-based residual fusion
        emb_tar = emb_tar.detach()
        z = torch.sigmoid(self.gate(torch.cat([emb_tar, emb_glo], dim=-1)))
        h_all = (1-z) * emb_tar + z * emb_glo  # [N, d]

        # (2) Edge-level lookup
        u_h = h_all[edge_index[0]]
        i_h = h_all[edge_index[1]]

        # (3) Gumbel keep/drop
        logits = self.mlp(torch.cat([u_h, i_h], dim=-1))
        keep_mask = F.gumbel_softmax(logits, tau=0.2, hard=True)[:, 0]

        # (4) cos × RBF reweighting
        u_norm = F.normalize(u_h, p=2, dim=1)
        i_norm = F.normalize(i_h, p=2, dim=1)
        cos_sim = (u_norm * i_norm).sum(dim=1)
        dist_sq = (u_h - i_h).pow(2).sum(dim=-1)
        rbf = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        sim_weight = 0.5 * (1 + cos_sim) * rbf

        return sim_weight, keep_mask


class RefinementModule(nn.Module):
    """
    Target-Guided Refinement Module (f1, Sec. 4.3).

    Operates in the auxiliary-observed environment, progressively pruning
    auxiliary relations misaligned with the target behavior via:
    (1) Target-guided auxiliary graph refiner (Sec. 4.3.1)
    (2) Target-guided preference learning (Sec. 4.3.2)
    """

    def __init__(self, data, emb_dim, gnn_layers, dataset, args):
        super(RefinementModule, self).__init__()
        self.dataset = dataset
        self.edge_dict = data['edge_dict']
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.emb_dim = emb_dim
        self.gnn_layers = gnn_layers
        self.bsg_types = data['bsg_types']
        self.total_behaviors = ['glo'] + self.bsg_types
        self.target_behavior = 'buy'
        self.aux_behaviors = [b for b in self.bsg_types if b != self.target_behavior]
        self.target_attn = TargetAttention(emb_dim, self.bsg_types)
        self.bpr_loss = BPRLoss()
        self.cl_temp = args.refine_cl_temp
        self.cl_w = args.refine_cl_w
        self.sigma = 20
        self.user_embedding = nn.Embedding(self.n_users + 1, emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, emb_dim, padding_idx=0)
        self.convs = nn.ModuleDict()
        for behavior_type in self.total_behaviors:
            self.convs[behavior_type] = nn.ModuleList([
                GraphConvLayer(emb_dim, emb_dim, 'gcn', sigma=self.sigma, dropout=0.1)
                for _ in range(self.gnn_layers)
            ])

        self.edge_pruners = nn.ModuleDict({
            behavior: GumbelEdgePruner(emb_dim, gumbel_temp=0.2, sigma=self.sigma)
            for behavior in self.aux_behaviors
        })

        self.refine_convs = nn.ModuleDict({
            behavior: nn.ModuleList([
                GraphConvLayer(emb_dim, emb_dim, 'dense', sigma=self.sigma, dropout=0.1)
                for _ in range(self.gnn_layers)
            ])
            for behavior in self.aux_behaviors
        })

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.target_attn.reset_parameters()

    def _propagate(self, x, edge_index, behavior_type):
        result = [x]
        for i, conv in enumerate(self.convs[behavior_type]):
            x = conv(x, edge_index)
            x = F.normalize(x, dim=-1)
            result.append(x / (i + 1))
        result = torch.stack(result, dim=0)
        return result.sum(dim=0)

    def _build_pruned_graph(self, behavior, emb_glo, emb_tar):
        edge_index = self.edge_dict[behavior]
        half_idx = edge_index.size(1) // 2
        ui_edges = edge_index[:, :half_idx]
        sim_weight, keep_mask = self.edge_pruners[behavior](emb_glo, emb_tar, ui_edges)
        selected = keep_mask.bool()
        filtered_ui = ui_edges[:, selected]
        filtered_w = sim_weight[selected]
        filtered_iu = torch.stack([filtered_ui[1], filtered_ui[0]], dim=0)
        refined_edge_index = torch.cat([filtered_ui, filtered_iu], dim=1)
        edge_weight = torch.cat([filtered_w, filtered_w], dim=0)
        return refined_edge_index, edge_weight
    
    def _propagate_refined(self, x, edge_index, edge_weight, behavior):
        result = [x]
        for i, conv in enumerate(self.refine_convs[behavior]):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.normalize(x, dim=-1)
            result.append(x / (i + 1))
        result = torch.stack(result, dim=0)
        return result.sum(dim=0)

    def forward(self):
        emb_dict = {}
        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        emb_glo = self._propagate(init_emb, self.edge_dict['glo'], 'glo')
        emb_target = self._propagate(emb_glo, self.edge_dict['buy'], 'buy')

        refined_aux_embs = []
        for behavior in self.aux_behaviors:
            if behavior in self.edge_dict:
                refined_edge_index, refined_edge_weight = self._build_pruned_graph(behavior, emb_glo, emb_target)
                emb_aux = self._propagate_refined(emb_glo, refined_edge_index, refined_edge_weight, behavior)
                # emb_aux = self._propagate(emb_glo, self.edge_dict[behavior], behavior)
                emb_dict[behavior] = emb_aux
                refined_aux_embs.append(emb_aux)

        emb_dict[self.target_behavior] = emb_target

        final_emb = self.target_attn(emb_dict)
        emb_dict['final'] = final_emb

        return final_emb, emb_target, refined_aux_embs


    def loss(self, user_indices, pos_indices, neg_indices):
        final_emb, emb_target, refined_aux_embs = self.forward()
        u_final, i_final = torch.split(final_emb, [self.n_users + 1, self.n_items + 1], dim=0)
        u_target, i_target = torch.split(emb_target, [self.n_users + 1, self.n_items + 1], dim=0)
        p_score = torch.einsum('ij,ij->i', u_final[user_indices], i_final[pos_indices])
        n_score = torch.einsum('ij,ij->i', u_final[user_indices], i_final[neg_indices])
        bpr_loss = self.bpr_loss(p_score, n_score)
        cl_loss = 0.0
        for emb_aux in refined_aux_embs:
            u_aux, i_aux = torch.split(emb_aux, [self.n_users + 1, self.n_items + 1], dim=0)
            cl_u = self._contrastive_loss(u_target[user_indices].detach(), u_aux[user_indices], self.cl_temp)
            cl_i = self._contrastive_loss(i_target[pos_indices].detach(), i_aux[pos_indices], self.cl_temp)
            cl_loss += (cl_u + cl_i)
        total_loss = bpr_loss + self.cl_w * cl_loss
        return total_loss

    @torch.no_grad()
    def precompute_embeddings(self):
        forward_output = self.forward()
        final_emb = forward_output[0]
        self._cached_user_emb, self._cached_item_emb = torch.split(
            final_emb, [self.n_users + 1, self.n_items + 1]
        )

    @torch.no_grad()
    def predict(self, users):
        user_emb = self._cached_user_emb[users.long()]
        return torch.mm(user_emb, self._cached_item_emb.t())

    def _contrastive_loss(self, x1, x2, temp):
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()
