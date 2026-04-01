"""
Learned Attention Mechanism for ARC-AGI-3
Selective focus on relevant parts of the grid
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AttentionMap:
    """Attention map over grid locations"""
    spatial_attention: np.ndarray  # [H, W] importance per cell
    object_attention: Dict[int, float]  # importance per object
    feature_attention: Dict[str, float]  # importance per feature type


class SpatialAttention(nn.Module):
    """
    Spatial attention: where to look in the grid
    """
    
    def __init__(self, grid_size: int = 64, num_heads: int = 4):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_heads = num_heads
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(64, 64, 1)
        self.key = nn.Conv2d(64, 64, 1)
        self.value = nn.Conv2d(64, 64, 1)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(64, num_heads, batch_first=True)
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention map
        
        Args:
            features: [B, 64, H, W] feature map
            
        Returns:
            [B, 1, H, W] attention weights
        """
        B, C, H, W = features.shape
        
        # Flatten spatial dimensions
        features_flat = features.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        
        # Self-attention
        q = self.query(features).view(B, C, H * W).transpose(1, 2)
        k = self.key(features).view(B, C, H * W).transpose(1, 2)
        v = self.value(features).view(B, C, H * W).transpose(1, 2)
        
        attn_out, _ = self.attention(q, k, v)  # [B, HW, C]
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        # Generate attention map
        attention_map = self.output_proj(attn_out)  # [B, 1, H, W]
        
        return attention_map


class ObjectAttention(nn.Module):
    """
    Object-level attention: which objects are important
    """
    
    def __init__(self, max_objects: int = 10, feature_dim: int = 64):
        super().__init__()
        
        self.max_objects = max_objects
        self.feature_dim = feature_dim
        
        # Object feature encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Attention scoring
        self.attention_scorer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Context encoder for global context
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim * max_objects, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, object_features: torch.Tensor,
               global_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute object attention scores
        
        Args:
            object_features: [B, N, D] features for N objects
            global_context: [B, D] global scene context
            
        Returns:
            [B, N] attention scores per object
        """
        B, N, D = object_features.shape
        
        # Encode objects
        encoded = self.object_encoder(object_features)  # [B, N, 64]
        
        # Add global context if available
        if global_context is not None:
            # Pad or truncate to max_objects
            padded_context = torch.zeros(B, self.max_objects, D)
            padded_context[:, :N, :] = object_features
            
            context_flat = padded_context.view(B, -1)
            context_vec = self.context_encoder(context_flat)  # [B, 64]
            
            # Broadcast to objects
            context_vec = context_vec.unsqueeze(1).expand(-1, N, -1)  # [B, N, 64]
            encoded = encoded + context_vec
        
        # Score each object
        scores = self.attention_scorer(encoded).squeeze(-1)  # [B, N]
        
        # Normalize
        scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
        
        return scores


class FeatureAttention(nn.Module):
    """
    Feature-level attention: which features to use
    """
    
    def __init__(self, num_features: int = 8):
        super().__init__()
        
        self.num_features = num_features
        
        # Learnable feature importance
        self.importance = nn.Parameter(torch.ones(num_features))
        
        # Context-dependent gating
        self.gate = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor,
               context: torch.Tensor) -> torch.Tensor:
        """
        Compute feature attention
        
        Args:
            features: [B, num_features, D] feature vectors
            context: [B, D] context vector
            
        Returns:
            [B, num_features] attention weights
        """
        B = features.shape[0]
        
        # Base importance
        base_importance = torch.softmax(self.importance, dim=0)  # [num_features]
        
        # Context-dependent modulation
        context_expanded = context.unsqueeze(1).expand(-1, self.num_features, -1)
        combined = torch.cat([features, context_expanded], dim=-1)
        
        # Global pooling
        combined_flat = combined.mean(dim=-1)  # [B, num_features]
        gate = self.gate(combined_flat)
        
        # Combine
        final_importance = base_importance * gate  # [B, num_features]
        
        return torch.softmax(final_importance, dim=-1)


class LearnedAttentionMechanism:
    """
    Complete learned attention system for ARC-AGI-3
    
    Integrates:
    - Spatial attention: where to look
    - Object attention: which objects matter
    - Feature attention: what to focus on
    """
    
    def __init__(self, grid_size: int = 64, device: str = 'cpu'):
        self.grid_size = grid_size
        self.device = device
        
        # Attention modules
        self.spatial_attn = SpatialAttention(grid_size).to(device)
        self.object_attn = ObjectAttention().to(device)
        self.feature_attn = FeatureAttention().to(device)
        
        # History for learning
        self.attention_history: List[AttentionMap] = []
        
    def compute_attention(self, grid: np.ndarray,
                         object_features: Optional[List[np.ndarray]] = None,
                         task_context: Optional[str] = None) -> AttentionMap:
        """
        Compute complete attention map for current state
        
        Args:
            grid: Current grid state [H, W]
            object_features: List of object feature vectors
            task_context: Description of current task
            
        Returns:
            AttentionMap with all attention components
        """
        # Convert to tensor
        grid_tensor = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Extract base features (simplified)
        base_features = self._extract_base_features(grid_tensor)
        
        # Spatial attention
        with torch.no_grad():
            spatial_map = self.spatial_attn(base_features)
        spatial_attention = spatial_map.squeeze().cpu().numpy()
        
        # Object attention
        object_attention = {}
        if object_features:
            obj_tensor = torch.FloatTensor(object_features).unsqueeze(0).to(self.device)
            global_feat = base_features.mean(dim=[2, 3])
            
            with torch.no_grad():
                obj_scores = self.object_attn(obj_tensor, global_feat)
            
            for i, score in enumerate(obj_scores.squeeze().cpu().numpy()):
                object_attention[i] = float(score)
        
        # Feature attention
        feature_attention = {
            'color': 0.3,
            'position': 0.3,
            'shape': 0.2,
            'relation': 0.2
        }
        
        if task_context:
            # Adjust based on task
            if 'color' in task_context.lower():
                feature_attention['color'] = 0.6
                feature_attention['position'] = 0.2
        
        attention_map = AttentionMap(
            spatial_attention=spatial_attention,
            object_attention=object_attention,
            feature_attention=feature_attention
        )
        
        self.attention_history.append(attention_map)
        
        return attention_map
    
    def apply_attention(self, grid: np.ndarray,
                       attention: AttentionMap) -> np.ndarray:
        """
        Apply attention to focus on relevant regions
        
        Returns:
            Attended grid (highlighted important regions)
        """
        # Weight grid by spatial attention
        attended = grid * (1 + attention.spatial_attention)
        
        return attended
    
    def get_top_k_regions(self, attention: AttentionMap,
                         k: int = 5) -> List[Tuple[int, int, float]]:
        """
        Get top-K most attended regions
        
        Returns:
            List of (y, x, attention_score)
        """
        attn = attention.spatial_attention
        
        # Flatten and get top K
        flat_attn = attn.flatten()
        top_k_indices = np.argsort(flat_attn)[-k:][::-1]
        
        regions = []
        for idx in top_k_indices:
            y = idx // attn.shape[1]
            x = idx % attn.shape[1]
            regions.append((y, x, flat_attn[idx]))
        
        return regions
    
    def _extract_base_features(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract base features from grid"""
        # Simple CNN feature extractor
        conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        ).to(self.device)
        
        with torch.no_grad():
            features = conv(grid)
        
        return features
    
    def learn_from_feedback(self, reward: float):
        """Update attention based on reward feedback"""
        # Increase attention to regions that led to success
        if reward > 0 and self.attention_history:
            recent = self.attention_history[-1]
            # Positive reinforcement
            pass
    
    def get_attention_stats(self) -> Dict[str, float]:
        """Get statistics about attention patterns"""
        if not self.attention_history:
            return {}
        
        recent = self.attention_history[-10:]
        
        spatial_entropy = np.mean([
            self._compute_entropy(a.spatial_attention)
            for a in recent
        ])
        
        return {
            'spatial_entropy': spatial_entropy,
            'num_objects_attended': np.mean([
                len(a.object_attention) for a in recent
            ]),
            'attention_focus': 1.0 - spatial_entropy  # Higher = more focused
        }
    
    def _compute_entropy(self, attention: np.ndarray) -> float:
        """Compute entropy of attention distribution"""
        flat = attention.flatten()
        flat = flat / (flat.sum() + 1e-8)
        
        entropy = -np.sum(flat * np.log(flat + 1e-8))
        
        # Normalize by max entropy
        max_entropy = np.log(len(flat))
        
        return entropy / max_entropy if max_entropy > 0 else 0


class SaliencyDetector:
    """
    Detect salient regions in grid (bottom-up attention)
    """
    
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        
    def detect_saliency(self, grid: np.ndarray) -> np.ndarray:
        """
        Detect salient regions based on:
        - Color contrast
        - Edge density
        - Uniqueness
        
        Returns:
            Saliency map [H, W]
        """
        saliency = np.zeros_like(grid, dtype=float)
        
        # Color uniqueness
        color_counts = np.bincount(grid.flatten(), minlength=16)
        rarity = 1.0 / (color_counts[grid] + 1)
        saliency += rarity * 0.3
        
        # Edge detection (gradient magnitude)
        gy, gx = np.gradient(grid.astype(float))
        edge_mag = np.sqrt(gx**2 + gy**2)
        saliency += edge_mag * 0.4
        
        # Center bias (objects in center often important)
        h, w = grid.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        center_bias = 1.0 - (dist_from_center / np.max(dist_from_center + 1))
        saliency += center_bias * 0.3
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
