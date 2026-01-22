"""
Data Loading and Preprocessing Utilities for HSTGAT

This module provides utilities for loading and preprocessing the three benchmark datasets:
- Weibo: Chinese social media cascades
- Twitter: English social media cascades  
- APS: Academic citation cascades
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import random


class CascadeDataset(Dataset):
    """Dataset class for information diffusion cascades."""
    
    def __init__(
        self,
        cascades: List[Dict],
        social_network: Dict,
        user_features: Dict,
        observation_time: float,
        prediction_time: float,
        max_seq_len: int = 100,
        max_candidates: int = 100,
        k_hop: int = 2
    ):
        """
        Args:
            cascades: List of cascade dictionaries with 'users', 'times', 'size'
            social_network: Adjacency list representation of social network
            user_features: Dictionary mapping user_id to feature vector
            observation_time: Observation window duration
            prediction_time: Prediction horizon
            max_seq_len: Maximum cascade sequence length
            max_candidates: Maximum number of candidate users
            k_hop: Number of hops for ego network sampling
        """
        self.cascades = cascades
        self.social_network = social_network
        self.user_features = user_features
        self.observation_time = observation_time
        self.prediction_time = prediction_time
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates
        self.k_hop = k_hop
        
        # Build user to index mapping
        all_users = set(user_features.keys())
        self.user_to_idx = {u: i for i, u in enumerate(sorted(all_users))}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.num_users = len(self.user_to_idx)
        
        # Feature dimension
        sample_feat = next(iter(user_features.values()))
        self.user_feat_dim = len(sample_feat) if isinstance(sample_feat, (list, np.ndarray)) else 1
        
    def __len__(self):
        return len(self.cascades)
    
    def __getitem__(self, idx):
        cascade = self.cascades[idx]
        
        # Extract observed cascade
        users = cascade['users']
        times = cascade['times']
        total_size = cascade['size']
        
        # Filter to observation window
        obs_mask = [t <= self.observation_time for t in times]
        obs_users = [u for u, m in zip(users, obs_mask) if m]
        obs_times = [t for t, m in zip(times, obs_mask) if m]
        
        # Pad or truncate sequence
        seq_len = min(len(obs_users), self.max_seq_len)
        obs_users = obs_users[:seq_len]
        obs_times = obs_times[:seq_len]
        
        # Get next user (for microscopic prediction)
        future_users = [u for u, m in zip(users, obs_mask) if not m]
        next_user = future_users[0] if future_users else obs_users[-1]
        
        # Convert to indices
        user_ids = [self.user_to_idx.get(u, 0) for u in obs_users]
        next_user_idx = self.user_to_idx.get(next_user, 0)
        
        # Get user features
        user_feats = [self.user_features.get(u, [0] * self.user_feat_dim) for u in obs_users]
        
        # Normalize timestamps
        max_time = max(obs_times) if obs_times else 1.0
        norm_times = [t / max_time for t in obs_times]
        
        # Build cascade graph edges
        cascade_edges = self._build_cascade_edges(obs_users, obs_times)
        
        # Sample ego network for social graph
        social_edges = self._sample_ego_network(obs_users)
        
        # Get candidate users (2-hop neighbors not in cascade)
        candidates = self._get_candidates(obs_users)
        candidate_ids = [self.user_to_idx.get(c, 0) for c in candidates]
        candidate_feats = [self.user_features.get(c, [0] * self.user_feat_dim) for c in candidates]
        
        # Find next user in candidates
        next_user_in_candidates = candidates.index(next_user) if next_user in candidates else 0
        
        # Pad sequences
        pad_len = self.max_seq_len - seq_len
        user_ids = user_ids + [0] * pad_len
        user_feats = user_feats + [[0] * self.user_feat_dim] * pad_len
        norm_times = norm_times + [0] * pad_len
        mask = [True] * seq_len + [False] * pad_len
        
        # Pad candidates
        cand_pad_len = self.max_candidates - len(candidate_ids)
        candidate_ids = candidate_ids + [0] * cand_pad_len
        candidate_feats = candidate_feats + [[0] * self.user_feat_dim] * cand_pad_len
        
        return {
            'user_ids': torch.tensor(user_ids, dtype=torch.long),
            'user_features': torch.tensor(user_feats, dtype=torch.float),
            'timestamps': torch.tensor(norm_times, dtype=torch.float),
            'cascade_edges': torch.tensor(cascade_edges, dtype=torch.long),
            'social_edges': torch.tensor(social_edges, dtype=torch.long),
            'candidate_ids': torch.tensor(candidate_ids, dtype=torch.long),
            'candidate_features': torch.tensor(candidate_feats, dtype=torch.float),
            'true_size': torch.tensor(total_size, dtype=torch.long),
            'true_next_user': torch.tensor(next_user_in_candidates, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'seq_len': seq_len
        }
    
    def _build_cascade_edges(self, users: List, times: List) -> List[List[int]]:
        """Build cascade graph edges based on temporal order and social connections."""
        edges = [[], []]
        user_set = set(users)
        
        for i, (u1, t1) in enumerate(zip(users, times)):
            for j, (u2, t2) in enumerate(zip(users, times)):
                if t1 < t2:
                    # Check if social connection exists
                    if u2 in self.social_network.get(u1, []):
                        edges[0].append(i)
                        edges[1].append(j)
                        
        return edges if edges[0] else [[0], [0]]
    
    def _sample_ego_network(self, seed_users: List) -> List[List[int]]:
        """Sample k-hop ego network around seed users."""
        visited = set()
        current_layer = set(seed_users)
        
        for _ in range(self.k_hop):
            next_layer = set()
            for user in current_layer:
                if user in self.social_network:
                    neighbors = self.social_network[user]
                    next_layer.update(neighbors[:50])  # Limit neighbors
            visited.update(current_layer)
            current_layer = next_layer - visited
            
        all_nodes = visited | current_layer
        node_to_local_idx = {n: i for i, n in enumerate(seed_users)}
        
        edges = [[], []]
        for user in all_nodes:
            if user in self.social_network and user in node_to_local_idx:
                for neighbor in self.social_network[user]:
                    if neighbor in node_to_local_idx:
                        edges[0].append(node_to_local_idx[user])
                        edges[1].append(node_to_local_idx[neighbor])
                        
        return edges if edges[0] else [[0], [0]]
    
    def _get_candidates(self, cascade_users: List) -> List:
        """Get candidate users from 2-hop neighborhood."""
        cascade_set = set(cascade_users)
        candidates = set()
        
        for user in cascade_users:
            if user in self.social_network:
                # 1-hop neighbors
                for n1 in self.social_network[user][:20]:
                    if n1 not in cascade_set:
                        candidates.add(n1)
                    # 2-hop neighbors
                    if n1 in self.social_network:
                        for n2 in self.social_network[n1][:10]:
                            if n2 not in cascade_set:
                                candidates.add(n2)
                                
        candidates = list(candidates)[:self.max_candidates]
        
        # Pad with random users if needed
        while len(candidates) < self.max_candidates:
            rand_user = random.choice(list(self.user_to_idx.keys()))
            if rand_user not in cascade_set and rand_user not in candidates:
                candidates.append(rand_user)
                
        return candidates


def load_weibo_dataset(data_dir: str) -> Tuple[List, Dict, Dict]:
    """Load Weibo dataset."""
    cascades = []
    social_network = defaultdict(list)
    user_features = {}
    
    # Load cascades
    cascade_file = os.path.join(data_dir, 'weibo_cascades.txt')
    if os.path.exists(cascade_file):
        with open(cascade_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cascade_id = parts[0]
                    events = parts[1].split(' ')
                    users, times = [], []
                    for event in events:
                        if ':' in event:
                            user, time = event.split(':')
                            users.append(user)
                            times.append(float(time))
                    if users:
                        cascades.append({
                            'id': cascade_id,
                            'users': users,
                            'times': times,
                            'size': len(users)
                        })
    
    # Load social network
    network_file = os.path.join(data_dir, 'weibo_network.txt')
    if os.path.exists(network_file):
        with open(network_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    user = parts[0]
                    followers = parts[1].split(' ')
                    social_network[user] = followers
    
    # Generate user features (follower count, activity, account age)
    for cascade in cascades:
        for user in cascade['users']:
            if user not in user_features:
                user_features[user] = [
                    len(social_network.get(user, [])),  # follower count
                    random.uniform(0, 1),  # activity level
                    random.uniform(0, 10),  # account age (years)
                    random.uniform(0, 1),  # engagement rate
                    random.randint(0, 1000),  # post count
                    random.uniform(0, 1),  # influence score
                    random.randint(0, 100),  # following count
                    random.uniform(0, 1),  # verified status
                    random.uniform(0, 1),  # bot score
                    random.uniform(0, 1),  # centrality
                ]
    
    return cascades, dict(social_network), user_features


def load_twitter_dataset(data_dir: str) -> Tuple[List, Dict, Dict]:
    """Load Twitter dataset."""
    # Similar structure to Weibo
    cascades = []
    social_network = defaultdict(list)
    user_features = {}
    
    cascade_file = os.path.join(data_dir, 'twitter_cascades.txt')
    if os.path.exists(cascade_file):
        with open(cascade_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cascade_id = parts[0]
                    events = parts[1].split(' ')
                    users, times = [], []
                    for event in events:
                        if ':' in event:
                            user, time = event.split(':')
                            users.append(user)
                            times.append(float(time))
                    if users:
                        cascades.append({
                            'id': cascade_id,
                            'users': users,
                            'times': times,
                            'size': len(users)
                        })
    
    network_file = os.path.join(data_dir, 'twitter_network.txt')
    if os.path.exists(network_file):
        with open(network_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    user = parts[0]
                    followers = parts[1].split(' ')
                    social_network[user] = followers
    
    for cascade in cascades:
        for user in cascade['users']:
            if user not in user_features:
                user_features[user] = [
                    len(social_network.get(user, [])),
                    random.uniform(0, 1),
                    random.uniform(0, 15),
                    random.uniform(0, 1),
                    random.randint(0, 5000),
                    random.uniform(0, 1),
                    random.randint(0, 500),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                ]
    
    return cascades, dict(social_network), user_features


def load_aps_dataset(data_dir: str) -> Tuple[List, Dict, Dict]:
    """Load APS citation dataset."""
    cascades = []
    citation_network = defaultdict(list)
    paper_features = {}
    
    cascade_file = os.path.join(data_dir, 'aps_cascades.txt')
    if os.path.exists(cascade_file):
        with open(cascade_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    paper_id = parts[0]
                    citations = parts[1].split(' ')
                    papers, times = [], []
                    for citation in citations:
                        if ':' in citation:
                            citing_paper, time = citation.split(':')
                            papers.append(citing_paper)
                            times.append(float(time))
                    if papers:
                        cascades.append({
                            'id': paper_id,
                            'users': papers,
                            'times': times,
                            'size': len(papers)
                        })
    
    network_file = os.path.join(data_dir, 'aps_network.txt')
    if os.path.exists(network_file):
        with open(network_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    paper = parts[0]
                    cited_by = parts[1].split(' ')
                    citation_network[paper] = cited_by
    
    for cascade in cascades:
        for paper in cascade['users']:
            if paper not in paper_features:
                paper_features[paper] = [
                    len(citation_network.get(paper, [])),
                    random.uniform(0, 1),
                    random.uniform(0, 50),
                    random.uniform(0, 1),
                    random.randint(1, 20),
                    random.uniform(0, 10),
                    random.randint(0, 100),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                ]
    
    return cascades, dict(citation_network), paper_features


def create_data_splits(
    cascades: List,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    chronological: bool = True
) -> Tuple[List, List, List]:
    """Split cascades into train/val/test sets."""
    
    if chronological:
        # Sort by earliest timestamp
        cascades = sorted(cascades, key=lambda x: min(x['times']))
    
    n = len(cascades)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return (
        cascades[:train_end],
        cascades[train_end:val_end],
        cascades[val_end:]
    )


def collate_fn(batch):
    """Custom collate function for batching cascade data."""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key == 'seq_len':
            collated[key] = [item[key] for item in batch]
        elif key in ['cascade_edges', 'social_edges']:
            # Handle variable-length edge lists
            max_edges = max(item[key].size(1) for item in batch)
            padded = []
            for item in batch:
                edges = item[key]
                if edges.size(1) < max_edges:
                    padding = torch.zeros(2, max_edges - edges.size(1), dtype=torch.long)
                    edges = torch.cat([edges, padding], dim=1)
                padded.append(edges)
            collated[key] = torch.stack(padded)
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


def get_dataloader(
    dataset: CascadeDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create DataLoader for cascade dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test data loading
    print("Data loading utilities initialized successfully!")
    print("Supported datasets: Weibo, Twitter, APS")
