"""
Training Script for HSTGAT

This script provides the complete training pipeline for the HSTGAT model,
including training loop, validation, early stopping, and checkpointing.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime

from model import HSTGAT, compute_loss
from data_utils import (
    CascadeDataset, 
    load_weibo_dataset, 
    load_twitter_dataset,
    load_aps_dataset,
    create_data_splits,
    get_dataloader
)
from metrics import (
    compute_msle, 
    compute_mape, 
    compute_hits_at_k, 
    compute_map_at_k
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_observation_time(dataset_name: str) -> float:
    """Get observation time window for each dataset."""
    obs_times = {
        'weibo': 3600,      # 1 hour in seconds
        'twitter': 7200,    # 2 hours in seconds
        'aps': 3 * 365 * 24 * 3600  # 3 years in seconds
    }
    return obs_times.get(dataset_name.lower(), 3600)


def get_prediction_time(dataset_name: str) -> float:
    """Get prediction horizon for each dataset."""
    pred_times = {
        'weibo': 24 * 3600,     # 24 hours
        'twitter': 48 * 3600,   # 48 hours
        'aps': 10 * 365 * 24 * 3600  # 10 years
    }
    return pred_times.get(dataset_name.lower(), 24 * 3600)


def train_epoch(model, dataloader, optimizer, device, mu=0.5):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_macro_loss = 0
    total_micro_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move batch to device
        user_ids = batch['user_ids'].to(device)
        user_features = batch['user_features'].to(device)
        timestamps = batch['timestamps'].to(device)
        cascade_edges = batch['cascade_edges'].to(device)
        social_edges = batch['social_edges'].to(device)
        candidate_ids = batch['candidate_ids'].to(device)
        candidate_features = batch['candidate_features'].to(device)
        true_size = batch['true_size'].to(device)
        true_next_user = batch['true_next_user'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        y_macro, scores, _ = model(
            user_ids=user_ids,
            user_features=user_features,
            timestamps=timestamps,
            cascade_edge_index=cascade_edges[0].T,  # Use first batch item's edges
            social_edge_index=social_edges[0].T,
            candidate_ids=candidate_ids,
            candidate_features=candidate_features,
            mask=mask
        )
        
        # Compute loss
        loss, macro_loss, micro_loss = compute_loss(
            y_macro, scores, true_size, true_next_user, mu=mu
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_macro_loss += macro_loss.item()
        total_micro_loss += micro_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'macro': f'{macro_loss.item():.4f}',
            'micro': f'{micro_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / num_batches,
        'macro_loss': total_macro_loss / num_batches,
        'micro_loss': total_micro_loss / num_batches
    }


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_pred_sizes = []
    all_true_sizes = []
    all_scores = []
    all_true_next = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            user_ids = batch['user_ids'].to(device)
            user_features = batch['user_features'].to(device)
            timestamps = batch['timestamps'].to(device)
            cascade_edges = batch['cascade_edges'].to(device)
            social_edges = batch['social_edges'].to(device)
            candidate_ids = batch['candidate_ids'].to(device)
            candidate_features = batch['candidate_features'].to(device)
            true_size = batch['true_size'].to(device)
            true_next_user = batch['true_next_user'].to(device)
            mask = batch['mask'].to(device)
            
            y_macro, scores, _ = model(
                user_ids=user_ids,
                user_features=user_features,
                timestamps=timestamps,
                cascade_edge_index=cascade_edges[0].T,
                social_edge_index=social_edges[0].T,
                candidate_ids=candidate_ids,
                candidate_features=candidate_features,
                mask=mask
            )
            
            # Convert predictions back from log scale
            pred_sizes = torch.exp(y_macro.squeeze()) - 1
            
            all_pred_sizes.extend(pred_sizes.cpu().numpy())
            all_true_sizes.extend(true_size.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            all_true_next.extend(true_next_user.cpu().numpy())
    
    all_pred_sizes = np.array(all_pred_sizes)
    all_true_sizes = np.array(all_true_sizes)
    all_scores = np.vstack(all_scores)
    all_true_next = np.array(all_true_next)
    
    # Compute metrics
    metrics = {
        'msle': compute_msle(all_true_sizes, all_pred_sizes),
        'mape': compute_mape(all_true_sizes, all_pred_sizes),
        'hits@5': compute_hits_at_k(all_scores, all_true_next, k=5),
        'hits@10': compute_hits_at_k(all_scores, all_true_next, k=10),
        'hits@20': compute_hits_at_k(all_scores, all_true_next, k=20),
        'hits@50': compute_hits_at_k(all_scores, all_true_next, k=50),
        'hits@100': compute_hits_at_k(all_scores, all_true_next, k=100),
        'map@10': compute_map_at_k(all_scores, all_true_next, k=10),
    }
    
    return metrics


def main(args):
    """Main training function."""
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset.lower() == 'weibo':
        cascades, social_network, user_features = load_weibo_dataset(args.data_dir)
    elif args.dataset.lower() == 'twitter':
        cascades, social_network, user_features = load_twitter_dataset(args.data_dir)
    elif args.dataset.lower() == 'aps':
        cascades, social_network, user_features = load_aps_dataset(args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Loaded {len(cascades)} cascades, {len(user_features)} users")
    
    # Create data splits
    train_cascades, val_cascades, test_cascades = create_data_splits(
        cascades, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15,
        chronological=True
    )
    
    print(f"Train: {len(train_cascades)}, Val: {len(val_cascades)}, Test: {len(test_cascades)}")
    
    # Get observation/prediction times
    obs_time = get_observation_time(args.dataset)
    pred_time = get_prediction_time(args.dataset)
    
    # Create datasets
    train_dataset = CascadeDataset(
        train_cascades, social_network, user_features,
        obs_time, pred_time, args.max_seq_len, args.max_candidates, args.k_hop
    )
    val_dataset = CascadeDataset(
        val_cascades, social_network, user_features,
        obs_time, pred_time, args.max_seq_len, args.max_candidates, args.k_hop
    )
    test_dataset = CascadeDataset(
        test_cascades, social_network, user_features,
        obs_time, pred_time, args.max_seq_len, args.max_candidates, args.k_hop
    )
    
    # Create dataloaders
    train_loader = get_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, args.batch_size, shuffle=False)
    
    # Initialize model
    model = HSTGAT(
        num_users=train_dataset.num_users,
        user_feat_dim=train_dataset.user_feat_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_temporal_layers=args.num_temporal_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        lambda1=args.lambda1,
        lambda2=args.lambda2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_msle = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.mu)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        print(f"Val MSLE: {val_metrics['msle']:.4f}, MAPE: {val_metrics['mape']:.2%}")
        print(f"Val Hits@10: {val_metrics['hits@10']:.4f}, MAP@10: {val_metrics['map@10']:.4f}")
        
        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Val/MSLE', val_metrics['msle'], epoch)
        writer.add_scalar('Val/MAPE', val_metrics['mape'], epoch)
        writer.add_scalar('Val/Hits@10', val_metrics['hits@10'], epoch)
        writer.add_scalar('Val/MAP@10', val_metrics['map@10'], epoch)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['msle'])
        
        # Early stopping check
        if val_metrics['msle'] < best_val_msle:
            best_val_msle = val_metrics['msle']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(output_dir, 'checkpoints', 'best_model.pt'))
            print(f"Saved best model with MSLE: {best_val_msle:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    checkpoint = torch.load(os.path.join(output_dir, 'checkpoints', 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  MSLE: {test_metrics['msle']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2%}")
    print(f"  Hits@5: {test_metrics['hits@5']:.4f}")
    print(f"  Hits@10: {test_metrics['hits@10']:.4f}")
    print(f"  Hits@20: {test_metrics['hits@20']:.4f}")
    print(f"  Hits@50: {test_metrics['hits@50']:.4f}")
    print(f"  Hits@100: {test_metrics['hits@100']:.4f}")
    print(f"  MAP@10: {test_metrics['map@10']:.4f}")
    
    # Save final results
    results = {
        'dataset': args.dataset,
        'seed': args.seed,
        'best_epoch': checkpoint['epoch'],
        'val_metrics': checkpoint['val_metrics'],
        'test_metrics': test_metrics,
        'args': vars(args)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    writer.close()
    print(f"\nResults saved to {output_dir}")
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HSTGAT model')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='weibo',
                        choices=['weibo', 'twitter', 'aps'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_gat_layers', type=int, default=2)
    parser.add_argument('--num_temporal_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--mu', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    # Data processing arguments
    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--max_candidates', type=int, default=100)
    parser.add_argument('--k_hop', type=int, default=2)
    
    args = parser.parse_args()
    main(args)
