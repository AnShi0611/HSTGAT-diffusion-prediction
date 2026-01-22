"""
HSTGAT: Hierarchical Spatial-Temporal Graph Attention Network
for Multi-Scale Information Diffusion Prediction in Social Networks

Author: An Shi
Email: ababa0611@163.com
"""

from .model import HSTGAT, compute_loss
from .metrics import (
    compute_msle,
    compute_mape,
    compute_hits_at_k,
    compute_map_at_k,
    compute_all_metrics
)
from .data_utils import (
    CascadeDataset,
    load_weibo_dataset,
    load_twitter_dataset,
    load_aps_dataset,
    create_data_splits,
    get_dataloader
)

__version__ = "1.0.0"
__author__ = "An Shi"
__email__ = "ababa0611@163.com"

__all__ = [
    "HSTGAT",
    "compute_loss",
    "compute_msle",
    "compute_mape",
    "compute_hits_at_k",
    "compute_map_at_k",
    "compute_all_metrics",
    "CascadeDataset",
    "load_weibo_dataset",
    "load_twitter_dataset",
    "load_aps_dataset",
    "create_data_splits",
    "get_dataloader",
]
