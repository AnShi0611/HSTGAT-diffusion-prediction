"""
HSTGAT: Hierarchical Spatial-Temporal Graph Attention Network
for Multi-Scale Information Diffusion Prediction in Social Networks

Author: An Shi
Affiliation: College of Media and Exhibition, Fujian Business University
Contact: ababa0611@163.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import math


class TemporalEncoding(nn.Module):
    """Sinusoidal temporal position encoding similar to Transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, timestamps):
        """
        Args:
            timestamps: Tensor of shape (batch_size, seq_len) with normalized timestamps
        Returns:
            Temporal encodings of shape (batch_size, seq_len, d_model)
        """
        # Convert continuous timestamps to indices
        indices = (timestamps * 1000).long().clamp(0, self.pe.size(0) - 1)
        return self.pe[indices]


class InputEmbedding(nn.Module):
    """Input Embedding Layer combining user ID, profile features, and temporal encoding."""
    
    def __init__(self, num_users, user_feat_dim, embed_dim, content_dim=768):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.profile_projection = nn.Linear(user_feat_dim, embed_dim)
        self.temporal_encoding = TemporalEncoding(embed_dim)
        self.content_projection = nn.Linear(content_dim, embed_dim)
        self.combine = nn.Linear(embed_dim * 3, embed_dim)
        
    def forward(self, user_ids, user_features, timestamps, content_embedding=None):
        """
        Args:
            user_ids: (batch_size, seq_len) user indices
            user_features: (batch_size, seq_len, user_feat_dim) profile features
            timestamps: (batch_size, seq_len) adoption timestamps (normalized)
            content_embedding: (batch_size, content_dim) optional content embedding
        Returns:
            Combined embeddings (batch_size, seq_len, embed_dim)
        """
        user_emb = self.user_embedding(user_ids)
        profile_emb = self.profile_projection(user_features)
        temporal_emb = self.temporal_encoding(timestamps)
        
        combined = torch.cat([user_emb, profile_emb, temporal_emb], dim=-1)
        output = self.combine(combined)
        
        if content_embedding is not None:
            content_emb = self.content_projection(content_embedding).unsqueeze(1)
            output = output + content_emb
            
        return output


class LocalCascadeGAT(nn.Module):
    """Local Cascade Channel - Graph Attention on cascade graph with temporal information."""
    
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.combine = nn.Linear(out_dim * 2, out_dim)
        
    def forward(self, x, edge_index, edge_time_diff=None):
        """
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge indices (2, num_edges)
            edge_time_diff: Time difference for each edge (num_edges, 1)
        Returns:
            Updated node features (num_nodes, out_dim)
        """
        gat_out = self.gat(x, edge_index)
        
        if edge_time_diff is not None:
            time_emb = self.time_encoder(edge_time_diff)
            # Aggregate time embeddings to nodes
            time_agg = torch.zeros_like(gat_out)
            time_agg.index_add_(0, edge_index[1], time_emb)
            gat_out = self.combine(torch.cat([gat_out, time_agg], dim=-1))
            
        return gat_out


class GlobalSocialGAT(nn.Module):
    """Global Social Channel - Graph Attention on social network with relationship encoding."""
    
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.rel_encoder = nn.Embedding(4, out_dim)  # 4 relationship types
        self.combine = nn.Linear(out_dim * 2, out_dim)
        
    def forward(self, x, edge_index, edge_type=None):
        """
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge indices (2, num_edges)
            edge_type: Relationship type for each edge (num_edges,)
        Returns:
            Updated node features (num_nodes, out_dim)
        """
        gat_out = self.gat(x, edge_index)
        
        if edge_type is not None:
            rel_emb = self.rel_encoder(edge_type)
            rel_agg = torch.zeros_like(gat_out)
            rel_agg.index_add_(0, edge_index[1], rel_emb)
            gat_out = self.combine(torch.cat([gat_out, rel_agg], dim=-1))
            
        return gat_out


class GatedFusion(nn.Module):
    """Gated Fusion mechanism to combine local and global channel outputs."""
    
    def __init__(self, hidden_dim, content_dim=None):
        super().__init__()
        input_dim = hidden_dim * 2
        if content_dim:
            input_dim += content_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, h_local, h_global, content_emb=None):
        """
        Args:
            h_local: Local channel output (num_nodes, hidden_dim)
            h_global: Global channel output (num_nodes, hidden_dim)
            content_emb: Optional content embedding (num_nodes, content_dim)
        Returns:
            Fused representation (num_nodes, hidden_dim)
        """
        if content_emb is not None:
            gate_input = torch.cat([h_local, h_global, content_emb], dim=-1)
        else:
            gate_input = torch.cat([h_local, h_global], dim=-1)
            
        g = self.gate(gate_input)
        return g * h_local + (1 - g) * h_global


class DualChannelGAT(nn.Module):
    """Dual-Channel Graph Attention Module combining local and global channels."""
    
    def __init__(self, in_dim, hidden_dim, heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_dim
            self.local_layers.append(LocalCascadeGAT(in_d, hidden_dim, heads, dropout))
            self.global_layers.append(GlobalSocialGAT(in_d, hidden_dim, heads, dropout))
            self.fusion_layers.append(GatedFusion(hidden_dim))
            
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, cascade_edge_index, social_edge_index, 
                edge_time_diff=None, edge_type=None):
        """
        Args:
            x: Input node features
            cascade_edge_index: Edges in cascade graph
            social_edge_index: Edges in social network (k-hop ego network)
            edge_time_diff: Time differences for cascade edges
            edge_type: Relationship types for social edges
        Returns:
            Fused node representations
        """
        h = x
        for local_layer, global_layer, fusion in zip(
            self.local_layers, self.global_layers, self.fusion_layers
        ):
            h_local = local_layer(h, cascade_edge_index, edge_time_diff)
            h_global = global_layer(h, social_edge_index, edge_type)
            h = fusion(h_local, h_global)
            h = self.layer_norm(h)
            
        return h


class DilatedCausalConv(nn.Module):
    """Dilated Causal Convolution layer for temporal modeling."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, channels)
        Returns:
            Output tensor (batch_size, seq_len, channels)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.conv(x)
        # Remove future padding for causal convolution
        out = out[:, :, :x.size(2)]
        out = out.transpose(1, 2)
        out = self.layer_norm(out)
        return F.relu(out)


class MultiScaleTemporalHierarchy(nn.Module):
    """Multi-Scale Temporal Hierarchy using dilated causal convolutions."""
    
    def __init__(self, hidden_dim, num_layers=4, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            dilation = 2 ** l
            self.layers.append(
                DilatedCausalConv(hidden_dim, hidden_dim, kernel_size, dilation)
            )
        
        # Scale attention
        self.scale_attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, H):
        """
        Args:
            H: Sequence of node representations (batch_size, seq_len, hidden_dim)
        Returns:
            Multi-scale temporal features (batch_size, seq_len, hidden_dim)
            Scale outputs list for analysis
        """
        scale_outputs = []
        h = H
        
        for layer in self.layers:
            h = layer(h) + h  # Residual connection
            scale_outputs.append(h)
            
        # Scale attention: combine outputs from all scales
        scale_stack = torch.stack(scale_outputs, dim=-1)  # (B, L, D, num_scales)
        attention_weights = F.softmax(
            self.scale_attention(scale_stack.permute(0, 1, 3, 2)).squeeze(-1), 
            dim=-1
        )  # (B, L, num_scales)
        
        Z = torch.sum(scale_stack * attention_weights.unsqueeze(2), dim=-1)
        
        return Z, scale_outputs


class MacroscopicHead(nn.Module):
    """Macroscopic Prediction Head for cascade size prediction."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, Z, mask=None):
        """
        Args:
            Z: Node representations (batch_size, seq_len, hidden_dim)
            mask: Optional mask for valid nodes
        Returns:
            Cascade size prediction (batch_size, 1)
            Cascade-level representation (batch_size, hidden_dim)
        """
        # Attention pooling
        attention_scores = self.attention(Z).squeeze(-1)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        c = torch.sum(Z * attention_weights.unsqueeze(-1), dim=1)
        y_macro = self.mlp(c)
        
        return y_macro, c


class MicroscopicHead(nn.Module):
    """Microscopic Prediction Head for next-user prediction."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, candidate_emb, Z, c):
        """
        Args:
            candidate_emb: Candidate user embeddings (batch_size, num_candidates, hidden_dim)
            Z: Cascade node representations (batch_size, seq_len, hidden_dim)
            c: Cascade-level representation (batch_size, hidden_dim)
        Returns:
            Candidate scores (batch_size, num_candidates)
        """
        # Attention over cascade for each candidate
        e_v_prime, _ = self.attention(candidate_emb, Z, Z)
        
        # Get last adopter representation
        z_last = Z[:, -1:, :].expand(-1, candidate_emb.size(1), -1)
        
        # Similarity with last adopter
        sim = F.cosine_similarity(candidate_emb, z_last, dim=-1).unsqueeze(-1)
        
        # Combine features
        c_expanded = c.unsqueeze(1).expand(-1, candidate_emb.size(1), -1)
        combined = torch.cat([e_v_prime, c_expanded, sim], dim=-1)
        
        scores = self.mlp(combined).squeeze(-1)
        
        return scores, e_v_prime


class CrossScaleBridge(nn.Module):
    """Cross-Scale Attention Bridge for bidirectional knowledge transfer."""
    
    def __init__(self, hidden_dim, lambda1=0.1, lambda2=0.1):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.macro_to_micro_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.micro_to_macro_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, scores, candidate_emb, c, y_macro):
        """
        Args:
            scores: Initial candidate scores (batch_size, num_candidates)
            candidate_emb: Candidate embeddings (batch_size, num_candidates, hidden_dim)
            c: Cascade-level representation (batch_size, hidden_dim)
            y_macro: Macroscopic prediction (batch_size, 1)
        Returns:
            Refined scores and cascade representation
        """
        # Macro-to-micro: augment scores with size prediction confidence
        c_expanded = c.unsqueeze(1).expand(-1, candidate_emb.size(1), -1)
        attn_out, _ = self.macro_to_micro_attn(candidate_emb, c_expanded, c_expanded)
        confidence = torch.sigmoid(y_macro)
        scores_refined = scores + self.lambda1 * attn_out.mean(dim=-1) * confidence
        
        # Micro-to-macro: refine cascade representation with predicted adopters
        adopter_weights = F.softmax(scores, dim=-1)
        weighted_candidates = torch.sum(adopter_weights.unsqueeze(-1) * candidate_emb, dim=1)
        c_refined = c + self.lambda2 * self.micro_to_macro_proj(weighted_candidates)
        
        return scores_refined, c_refined


class HSTGAT(nn.Module):
    """
    HSTGAT: Hierarchical Spatial-Temporal Graph Attention Network
    for Multi-Scale Information Diffusion Prediction
    """
    
    def __init__(
        self,
        num_users,
        user_feat_dim=10,
        embed_dim=128,
        hidden_dim=128,
        num_gat_layers=2,
        num_temporal_layers=4,
        num_heads=4,
        dropout=0.1,
        lambda1=0.1,
        lambda2=0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Input Embedding
        self.input_embedding = InputEmbedding(num_users, user_feat_dim, embed_dim)
        
        # Dual-Channel Graph Attention
        self.dual_channel_gat = DualChannelGAT(
            embed_dim, hidden_dim, num_heads, num_gat_layers, dropout
        )
        
        # Multi-Scale Temporal Hierarchy
        self.temporal_hierarchy = MultiScaleTemporalHierarchy(
            hidden_dim, num_temporal_layers
        )
        
        # Prediction Heads
        self.macro_head = MacroscopicHead(hidden_dim)
        self.micro_head = MicroscopicHead(hidden_dim)
        
        # Cross-Scale Bridge
        self.cross_scale_bridge = CrossScaleBridge(hidden_dim, lambda1, lambda2)
        
    def forward(
        self,
        user_ids,
        user_features,
        timestamps,
        cascade_edge_index,
        social_edge_index,
        candidate_ids,
        candidate_features,
        edge_time_diff=None,
        edge_type=None,
        content_embedding=None,
        mask=None
    ):
        """
        Forward pass of HSTGAT.
        
        Args:
            user_ids: (batch_size, seq_len) user indices in cascade
            user_features: (batch_size, seq_len, user_feat_dim) user profile features
            timestamps: (batch_size, seq_len) adoption timestamps
            cascade_edge_index: (2, num_cascade_edges) cascade graph edges
            social_edge_index: (2, num_social_edges) social network edges
            candidate_ids: (batch_size, num_candidates) candidate user indices
            candidate_features: (batch_size, num_candidates, user_feat_dim)
            edge_time_diff: Optional time differences for cascade edges
            edge_type: Optional relationship types for social edges
            content_embedding: Optional content embedding
            mask: Optional mask for valid positions
            
        Returns:
            y_macro: Cascade size prediction (batch_size, 1)
            scores: Candidate user scores (batch_size, num_candidates)
            c: Cascade representation (batch_size, hidden_dim)
        """
        batch_size, seq_len = user_ids.shape
        
        # Input Embedding
        E = self.input_embedding(user_ids, user_features, timestamps, content_embedding)
        
        # Flatten for graph processing
        E_flat = E.view(-1, self.embed_dim)
        
        # Dual-Channel Graph Attention
        H = self.dual_channel_gat(
            E_flat, cascade_edge_index, social_edge_index,
            edge_time_diff, edge_type
        )
        
        # Reshape back to batch
        H = H.view(batch_size, seq_len, self.hidden_dim)
        
        # Multi-Scale Temporal Hierarchy
        Z, scale_outputs = self.temporal_hierarchy(H)
        
        # Macroscopic Prediction
        y_macro, c = self.macro_head(Z, mask)
        
        # Candidate embeddings
        candidate_emb = self.input_embedding.user_embedding(candidate_ids)
        
        # Microscopic Prediction
        scores, _ = self.micro_head(candidate_emb, Z, c)
        
        # Cross-Scale Bridge
        scores_refined, c_refined = self.cross_scale_bridge(scores, candidate_emb, c, y_macro)
        
        return y_macro, scores_refined, c_refined


def compute_loss(y_macro, scores, true_size, true_next_user, mu=0.5, margin=0.5):
    """
    Compute combined loss for HSTGAT.
    
    Args:
        y_macro: Predicted cascade size (batch_size, 1)
        scores: Candidate scores (batch_size, num_candidates)
        true_size: Ground truth cascade size (batch_size,)
        true_next_user: Index of true next user in candidates (batch_size,)
        mu: Task balance weight
        margin: Margin for ranking loss
        
    Returns:
        Total loss, macro loss, micro loss
    """
    # Macroscopic loss: MSE on log-transformed sizes
    log_true_size = torch.log(true_size.float() + 1)
    loss_macro = F.mse_loss(y_macro.squeeze(), log_true_size)
    
    # Microscopic loss: Cross-entropy + margin ranking
    loss_ce = F.cross_entropy(scores, true_next_user)
    
    # Margin ranking loss
    batch_size = scores.size(0)
    pos_scores = scores.gather(1, true_next_user.unsqueeze(1))
    neg_mask = torch.ones_like(scores, dtype=torch.bool)
    neg_mask.scatter_(1, true_next_user.unsqueeze(1), False)
    neg_scores = scores[neg_mask].view(batch_size, -1)
    
    # Sample negatives
    num_neg_samples = min(10, neg_scores.size(1))
    neg_indices = torch.randint(0, neg_scores.size(1), (batch_size, num_neg_samples), device=scores.device)
    sampled_neg = neg_scores.gather(1, neg_indices)
    
    loss_rank = F.relu(margin - pos_scores + sampled_neg).mean()
    
    loss_micro = loss_ce + loss_rank
    
    # Combined loss
    total_loss = loss_macro + mu * loss_micro
    
    return total_loss, loss_macro, loss_micro


if __name__ == "__main__":
    # Test model initialization
    model = HSTGAT(
        num_users=10000,
        user_feat_dim=10,
        embed_dim=128,
        hidden_dim=128
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("HSTGAT model initialized successfully!")
