import os
import torch
import random
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class BPRDataset(Dataset):
    def __init__(self, buy_interactions, n_items):
        self.buy_interactions = {int(k): set(v) for k, v in buy_interactions.items()}
        self.all_items = set(range(1, n_items + 1))
        self.users = list(self.buy_interactions.keys())
        self.total_samples = []

        for user in self.users:
            for pos_item in self.buy_interactions[user]:
                self.total_samples.append((user, pos_item))

    def __len__(self):
        return len(self.total_samples)

    def __getitem__(self, idx):
        user, pos_item = self.total_samples[idx]
        neg_candidates = list(self.all_items - self.buy_interactions[user])
        neg_idx = torch.randint(0, len(neg_candidates), (1,)).item()
        neg_item = neg_candidates[neg_idx]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


def convert_edge(edge_list, n_users):
    """Convert edge list to bidirectional graph with item index offset."""
    edge_list[1] += n_users + 1
    edge_list = torch.cat([edge_list, edge_list.flip(0)], dim=1)
    return edge_list


def seed_worker(worker_id):
    """Ensure reproducibility in DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data(data_dir, dataset, device, batch_size, seed=42):
    logger.info('Loading data')
    data_dir = os.path.join(data_dir, dataset)

    # Load data statistics
    with open(f'{data_dir}/statistics.json', 'r') as f:
        statistics = json.load(f)

    n_users, n_items = statistics['n_users'], statistics['n_items']
    bsg_types = statistics['bsg_types']

    # Load edges
    edge_dict = dict()
    for behavior_type in ['glo', 'aux_ob'] + bsg_types:
        edge_list = np.loadtxt(f'{data_dir}/{behavior_type}.txt', dtype=int)
        edge_list = torch.from_numpy(edge_list).to(device).T
        edge_dict[behavior_type] = convert_edge(edge_list, n_users)

    # Load train/test buy interactions
    with open(f'{data_dir}/train.json', 'r') as f:
        train_buy = json.load(f)
    with open(f'{data_dir}/test.json', 'r') as f:
        test_buy = json.load(f)

    train_dataset = BPRDataset(train_buy, n_items)
    test_dataset = BPRDataset(test_buy, n_items)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=8,
        worker_init_fn=seed_worker, generator=g,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=8,
        worker_init_fn=seed_worker, generator=g,
    )
    test_gt_length = np.array([len(items) for items in test_buy.values()])

    # Load test_observed and test_unobserved splits
    test_observed_buy = None
    test_unobserved_buy = None
    test_observed_path = f'{data_dir}/test_observed.json'
    test_unobserved_path = f'{data_dir}/test_unobserved.json'

    if os.path.exists(test_observed_path) and os.path.exists(test_unobserved_path):
        logger.info('Loading test_observed and test_unobserved data')
        with open(test_observed_path, 'r') as f:
            test_observed_buy = json.load(f)
        with open(test_unobserved_path, 'r') as f:
            test_unobserved_buy = json.load(f)
    else:
        logger.warning(f'test_observed.json or test_unobserved.json not found in {data_dir}.')

    test_observed_dataset = BPRDataset(test_observed_buy, n_items)
    test_unobserved_dataset = BPRDataset(test_unobserved_buy, n_items)
    test_observed_loader = DataLoader(
        test_observed_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=8,
        worker_init_fn=seed_worker, generator=g,
    )
    test_unobserved_loader = DataLoader(
        test_unobserved_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=8,
        worker_init_fn=seed_worker, generator=g,
    )

    data = dict()
    data['edge_dict'] = edge_dict
    data['train_loader'] = train_loader
    data['test_loader'] = test_loader
    data['test_observed_loader'] = test_observed_loader
    data['test_unobserved_loader'] = test_unobserved_loader
    data['train_gt'] = train_buy
    data['test_gt'] = test_buy
    data['test_gt_length'] = test_gt_length

    if test_observed_buy is not None:
        data['test_observed_gt'] = test_observed_buy
    if test_unobserved_buy is not None:
        data['test_unobserved_gt'] = test_unobserved_buy

    data.update(statistics)
    return data
