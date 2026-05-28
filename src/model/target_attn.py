import torch
import torch.nn as nn


class TargetAttention(nn.Module):
    def __init__(self, emb_dim, bsg_types):
        super(TargetAttention, self).__init__()
        self.emb_dim = emb_dim
        self.bsg_types = bsg_types
        self.lin = nn.Linear(emb_dim * 2, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, emb_dict):
        key = emb_dict['buy']
        key = key.unsqueeze(1).repeat(1, len(self.bsg_types), 1)
        query_emb = torch.stack([emb_dict[b] for b in self.bsg_types], dim=1)
        concat_emb = torch.cat([key, query_emb], dim=2)
        attention = self.lin(concat_emb).softmax(dim=1)
        updated_emb = (attention * query_emb).sum(dim=1)
        return updated_emb
