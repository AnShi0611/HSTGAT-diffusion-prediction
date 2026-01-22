"""
Evaluation Metrics for HSTGAT

This module provides metrics for evaluating both macroscopic and microscopic
information diffusion prediction tasks.
"""

import numpy as np
from typing import List, Union, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def compute_msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Logarithmic Error (MSLE).
    
    MSLE = (1/n) * Σ(log(y_pred + 1) - log(y_true + 1))²
    
    Args:
        y_true: Ground truth cascade sizes
        y_pred: Predicted cascade sizes
        
    Returns:
        MSLE value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Ensure non-negative
    y_pred = np.maximum(y_pred, 0)
    
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    return np.mean((log_pred - log_true) ** 2)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).
    
    MAPE = (1/n) * Σ|y_true - y_pred| / max(y_true, epsilon)
    
    Args:
        y_true: Ground truth cascade sizes
        y_pred: Predicted cascade sizes
        epsilon: Small constant to avoid division by zero
        
    Returns:
        MAPE value (as decimal, not percentage)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Ensure non-negative
    y_pred = np.maximum(y_pred, 0)
    
    # Avoid division by zero
    denominator = np.maximum(y_true, epsilon)
    
    return np.mean(np.abs(y_true - y_pred) / denominator)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE).
    
    Args:
        y_true: Ground truth cascade sizes
        y_pred: Predicted cascade sizes
        
    Returns:
        RMSE value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE).
    
    Args:
        y_true: Ground truth cascade sizes
        y_pred: Predicted cascade sizes
        
    Returns:
        MAE value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    return np.mean(np.abs(y_true - y_pred))


def compute_hits_at_k(scores: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """
    Compute Hits@K for microscopic prediction.
    
    Hits@K measures the proportion of times the true next user
    appears in the top-K predicted candidates.
    
    Args:
        scores: Predicted scores for each candidate (n_samples, n_candidates)
        targets: True next user index for each sample (n_samples,)
        k: Number of top candidates to consider
        
    Returns:
        Hits@K value
    """
    scores = np.asarray(scores)
    targets = np.asarray(targets).flatten()
    
    n_samples = scores.shape[0]
    hits = 0
    
    for i in range(n_samples):
        # Get top-k predictions
        top_k_indices = np.argsort(scores[i])[::-1][:k]
        if targets[i] in top_k_indices:
            hits += 1
            
    return hits / n_samples


def compute_map_at_k(scores: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """
    Compute Mean Average Precision at K (MAP@K).
    
    Args:
        scores: Predicted scores for each candidate (n_samples, n_candidates)
        targets: True next user index for each sample (n_samples,)
        k: Number of top candidates to consider
        
    Returns:
        MAP@K value
    """
    scores = np.asarray(scores)
    targets = np.asarray(targets).flatten()
    
    n_samples = scores.shape[0]
    ap_sum = 0
    
    for i in range(n_samples):
        # Get ranking
        ranked_indices = np.argsort(scores[i])[::-1][:k]
        
        # Find position of true target
        if targets[i] in ranked_indices:
            position = np.where(ranked_indices == targets[i])[0][0] + 1
            ap_sum += 1.0 / position
            
    return ap_sum / n_samples


def compute_mrr(scores: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        scores: Predicted scores for each candidate (n_samples, n_candidates)
        targets: True next user index for each sample (n_samples,)
        
    Returns:
        MRR value
    """
    scores = np.asarray(scores)
    targets = np.asarray(targets).flatten()
    
    n_samples = scores.shape[0]
    rr_sum = 0
    
    for i in range(n_samples):
        # Get full ranking
        ranked_indices = np.argsort(scores[i])[::-1]
        
        # Find position of true target
        if targets[i] in ranked_indices:
            position = np.where(ranked_indices == targets[i])[0][0] + 1
            rr_sum += 1.0 / position
            
    return rr_sum / n_samples


def compute_ndcg_at_k(scores: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K (NDCG@K).
    
    Args:
        scores: Predicted scores for each candidate (n_samples, n_candidates)
        targets: True next user index for each sample (n_samples,)
        k: Number of top candidates to consider
        
    Returns:
        NDCG@K value
    """
    scores = np.asarray(scores)
    targets = np.asarray(targets).flatten()
    
    n_samples = scores.shape[0]
    ndcg_sum = 0
    
    for i in range(n_samples):
        # Get top-k predictions
        ranked_indices = np.argsort(scores[i])[::-1][:k]
        
        # Compute DCG
        dcg = 0
        for j, idx in enumerate(ranked_indices):
            if idx == targets[i]:
                dcg = 1.0 / np.log2(j + 2)  # +2 because positions are 1-indexed
                break
        
        # Ideal DCG (target at position 1)
        idcg = 1.0 / np.log2(2)
        
        ndcg_sum += dcg / idcg
        
    return ndcg_sum / n_samples


def compute_all_metrics(
    pred_sizes: np.ndarray,
    true_sizes: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
    k_values: List[int] = [5, 10, 20, 50, 100]
) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        pred_sizes: Predicted cascade sizes
        true_sizes: Ground truth cascade sizes
        scores: Candidate scores for microscopic prediction
        targets: True next user indices
        k_values: List of K values for Hits@K and MAP@K
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        # Macroscopic metrics
        'msle': compute_msle(true_sizes, pred_sizes),
        'mape': compute_mape(true_sizes, pred_sizes),
        'rmse': compute_rmse(true_sizes, pred_sizes),
        'mae': compute_mae(true_sizes, pred_sizes),
        
        # Microscopic metrics
        'mrr': compute_mrr(scores, targets),
    }
    
    # Add Hits@K and MAP@K for various K values
    for k in k_values:
        metrics[f'hits@{k}'] = compute_hits_at_k(scores, targets, k)
        metrics[f'map@{k}'] = compute_map_at_k(scores, targets, k)
        metrics[f'ndcg@{k}'] = compute_ndcg_at_k(scores, targets, k)
    
    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    print("\nMacroscopic Prediction:")
    print(f"  MSLE:  {metrics['msle']:.4f}")
    print(f"  MAPE:  {metrics['mape']:.2%}")
    print(f"  RMSE:  {metrics['rmse']:.4f}")
    print(f"  MAE:   {metrics['mae']:.4f}")
    
    print("\nMicroscopic Prediction:")
    print(f"  MRR:   {metrics['mrr']:.4f}")
    
    # Print Hits@K
    hits_keys = sorted([k for k in metrics.keys() if k.startswith('hits@')])
    if hits_keys:
        print(f"  Hits@K:", end="")
        for k in hits_keys:
            val = int(k.split('@')[1])
            print(f" @{val}={metrics[k]:.4f}", end="")
        print()
    
    # Print MAP@K
    map_keys = sorted([k for k in metrics.keys() if k.startswith('map@')])
    if map_keys:
        print(f"  MAP@K:", end="")
        for k in map_keys:
            val = int(k.split('@')[1])
            print(f" @{val}={metrics[k]:.4f}", end="")
        print()
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test metrics with dummy data
    np.random.seed(42)
    n_samples = 100
    n_candidates = 50
    
    # Generate test data
    true_sizes = np.random.exponential(50, n_samples)
    pred_sizes = true_sizes + np.random.normal(0, 10, n_samples)
    
    scores = np.random.randn(n_samples, n_candidates)
    targets = np.random.randint(0, n_candidates, n_samples)
    
    # Compute metrics
    metrics = compute_all_metrics(pred_sizes, true_sizes, scores, targets)
    print_metrics(metrics, "Test Metrics")
