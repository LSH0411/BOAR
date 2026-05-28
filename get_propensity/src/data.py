import os
import torch
import random
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader 
from loguru import logger

class AuxObDataset(Dataset):
    """Dataset for aux_ob interactions"""
    def __init__(self, aux_ob_interactions, n_items, neg_sample):
        self.aux_ob_interactions = {int(k): set(v) for k, v in aux_ob_interactions.items()}
        self.all_items = set(range(1, n_items + 1))
        self.users = list(self.aux_ob_interactions.keys())
        self.neg_sample = neg_sample
        self.total_samples = []

        for user in self.users:
            for pos_item in self.aux_ob_interactions[user]:
                self.total_samples.append((user, pos_item))

    def __len__(self):
        return len(self.total_samples)

    def __getitem__(self, idx):
        user, pos_item = self.total_samples[idx]
        neg_candidates = list(self.all_items - self.aux_ob_interactions[user])
        neg_items = random.sample(neg_candidates, self.neg_sample)

        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
                torch.tensor(neg_items, dtype=torch.long))

def convert_edge(edge_list, n_users):
    # Item index starts from n_users+1
    edge_list[1] += n_users + 1
    # Add reverse item to user edges 
    edge_list = torch.cat([edge_list, edge_list.flip(0)], dim=1)
    return edge_list


def extract_aux_ob_interactions(data_dir):
    """Extract aux_ob interactions from aux_ob.txt"""
    aux_ob_path = f'{data_dir}/aux_ob.txt'

    if not os.path.exists(aux_ob_path):
        logger.error(f'aux_ob.txt not found in {data_dir}')
        raise FileNotFoundError(f'aux_ob.txt not found in {data_dir}')

    aux_ob_edges = np.loadtxt(aux_ob_path, dtype=int)

    aux_ob_interactions = {}
    for user, item in aux_ob_edges:
        if user not in aux_ob_interactions:
            aux_ob_interactions[user] = []
        aux_ob_interactions[user].append(item)

    aux_ob_interactions = {str(k): v for k, v in aux_ob_interactions.items()}

    logger.info(f'Extracted aux_ob interactions: {len(aux_ob_interactions)} users, {len(aux_ob_edges)} interactions')

    return aux_ob_interactions

    
def load_data(data_dir, dataset, device, batch_size, neg_sample=1):
    logger.info('Load data for Propensity Network')
    data_dir = os.path.join(data_dir, dataset)
    
    # load data statistics
    with open(f'{data_dir}/statistics.json', 'r') as f:
        statistics = json.load(f)
        
    n_users, n_items = statistics['n_users'], statistics['n_items']    
    # load edges (only aux_ob needed for propensity network)
    edge_dict = dict()
    for behavior_type in ['aux_ob', 'glo', 'buy']:
        edge_path = f'{data_dir}/{behavior_type}.txt'
        if os.path.exists(edge_path):
            edge_list = np.loadtxt(edge_path, dtype=int)
            edge_list = torch.from_numpy(edge_list).to(device).T
            edge_dict[behavior_type] = convert_edge(edge_list, n_users)

    # Extract aux_ob interactions for training
    aux_ob_interactions = extract_aux_ob_interactions(data_dir)
    aux_ob_train_dataset = AuxObDataset(aux_ob_interactions, n_items, neg_sample)
    aux_ob_train_loader = DataLoader(
        aux_ob_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    logger.info(f'aux_ob train dataset: {len(aux_ob_train_dataset)} samples')
    data = dict()
    data['edge_dict'] = edge_dict
    data['aux_ob_train_loader'] = aux_ob_train_loader
    data['aux_ob_interactions'] = aux_ob_interactions
    data['n_users'] = n_users
    data['n_items'] = n_items
    
    return data