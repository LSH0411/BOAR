import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .graph_conv import GraphConvLayer


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, scores, labels):
        return self.bce(scores, labels)

class LGCN_G(nn.Module):
    def __init__(self, data, emb_dim, gnn_layers, dataset, args):
        super(LGCN_G, self).__init__()
        self.dataset = dataset
        
        self.edge_dict = data['edge_dict']
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.emb_dim = emb_dim
        self.gnn_layers = gnn_layers        
    
        self.bce_loss = BCELoss()
        
        self.neg_sample = args.neg_sample
        
        
        self.user_embedding = nn.Embedding(self.n_users+1, emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items+1, emb_dim, padding_idx=0)
        
        # Graph conv layers
        self.convs = nn.ModuleDict()

        self.convs['aux_ob'] = nn.ModuleList([GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(self.gnn_layers)])
                
        self.reset_parameters()

    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    
    def propagate(self, x, edge_index, behavior_type, target_emb=None):
        result = [x]
        for i, conv in enumerate(self.convs[behavior_type]):
            x = conv(x, edge_index, target_emb)
            x = F.normalize(x, dim=-1)
            result.append(x/(i+1))
        result = torch.stack(result, dim=0)
        x = result.sum(dim=0)
        return x
    
    def forward(self):
        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        emb_aux_ob = self.propagate(init_emb, self.edge_dict['aux_ob'], 'aux_ob')

        return emb_aux_ob


    def loss(self, users, pos_items, neg_items):
        """Compute BCE loss for propensity network training.
        neg_items: (B, neg_sample)
        """
        aux_ob_emb = self.forward()
        user_emb, item_emb = torch.split(aux_ob_emb, [self.n_users + 1, self.n_items + 1], dim=0)

        pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)  # (B,)

        # neg_items: (B, K) → flatten to (B*K,)
        B, K = neg_items.shape
        neg_items_flat = neg_items.reshape(-1)
        users_expanded = users.repeat_interleave(K)
        neg_scores = (user_emb[users_expanded] * item_emb[neg_items_flat]).sum(dim=1)  # (B*K,)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

        return self.bce_loss(scores, labels)
    
    def predict(self, users):
        """Predict propensity scores for given users"""
        aux_ob_emb = self.forward()
        user_emb, item_emb = torch.split(aux_ob_emb, [self.n_users + 1, self.n_items + 1], dim=0)
        
        user_emb = user_emb[users.long()]
        scores = torch.matmul(user_emb, item_emb.transpose(0, 1))
        
        # Apply sigmoid to get propensity in [0, 1]
        propensity = torch.sigmoid(scores)
        
        return propensity
    