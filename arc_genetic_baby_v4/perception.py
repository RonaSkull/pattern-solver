"""Layer 1: Predictive Perception - Hierarchical Predictive Coding.

Implements Predictive Processing theory: brain as a prediction machine.
Based on Friston's Free Energy Principle and hierarchical predictive coding.

References:
    - Friston, K. (2010). The free-energy principle: a unified brain theory?
    - Clark, A. (2013). Whatever next? Predictive brains, situated agents, 
      and the future of cognitive science.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .config import PerceptionConfig


@dataclass
class BeliefState:
    """Represents hierarchical beliefs about the current state."""
    level1_features: np.ndarray  # Low-level: pixel/color predictions
    level2_patterns: np.ndarray  # Mid-level: spatial patterns
    level3_structure: np.ndarray  # High-level: structural/object beliefs
    prediction_errors: List[float]  # Free Energy at each level
    confidence: float  # Overall confidence in beliefs
    
    def total_free_energy(self) -> float:
        """Total Free Energy across all levels."""
        return sum(self.prediction_errors)


class PredictiveLayer(nn.Module, ABC):
    """Abstract base for a predictive coding layer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
    @abstractmethod
    def forward(self, bottom_up: torch.Tensor, top_down: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining bottom-up and top-down signals.
        
        Returns:
            prediction: Top-down prediction
            prediction_error: Difference between input and prediction
        """
        pass
    
    @abstractmethod
    def update(self, prediction_error: torch.Tensor, lr: float = 0.001):
        """Update layer weights based on prediction error (learning)."""
        pass


class Level1PredictiveLayer(PredictiveLayer):
    """
    Level 1: Pixel/Color prediction.
    Predicts changes in individual pixels and colors.
    """
    
    def __init__(self, grid_size: int = 64, num_colors: int = 16, hidden_dim: int = 128):
        # Use fixed input dim for flexibility - will adapt to actual input
        super().__init__(grid_size * grid_size, hidden_dim, grid_size * grid_size)
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.max_input_dim = grid_size * grid_size
        
        # Encoder: bottom-up processing
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Projection layer to ensure consistent output size for next level
        self.encoder_projection = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        
        # Decoder: top-down prediction generation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, bottom_up: torch.Tensor, top_down: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input grid and generate predictions.
        Handles variable input sizes by padding/truncating to expected dimensions.
        """
        # Adapt input size to match encoder's expected input
        batch_size = bottom_up.shape[0]
        actual_dim = bottom_up.shape[1]
        
        if actual_dim != self.input_dim:
            # Pad or truncate to match expected input dimension
            if actual_dim < self.input_dim:
                # Pad with zeros
                padding = torch.zeros(batch_size, self.input_dim - actual_dim, 
                                     device=bottom_up.device)
                bottom_up_adjusted = torch.cat([bottom_up, padding], dim=1)
            else:
                # Truncate
                bottom_up_adjusted = bottom_up[:, :self.input_dim]
        else:
            bottom_up_adjusted = bottom_up
        
        # Encode input
        encoded = self.encoder(bottom_up_adjusted)
        
        # If top-down signal exists, incorporate it
        if top_down is not None:
            # Combine encoded input with top-down predictions
            encoded = encoded + 0.1 * top_down  # Small top-down influence
        
        # Generate prediction
        prediction = self.decoder(encoded)
        
        # Calculate prediction error (Free Energy signal at this level)
        prediction_error = torch.abs(bottom_up_adjusted - prediction)
        
        return prediction, prediction_error
    
    def update(self, prediction_error: torch.Tensor, lr: float = 0.001):
        """Update using prediction error as loss."""
        loss = prediction_error.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Level2PredictiveLayer(PredictiveLayer):
    """
    Level 2: Spatial pattern prediction.
    Predicts patterns like symmetry, rotation, translation, shape properties.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__(input_dim, hidden_dim, input_dim)
        
        # Pattern detection networks
        self.pattern_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.pattern_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Learned pattern prototypes
        self.pattern_prototypes = nn.Parameter(torch.randn(16, hidden_dim // 2))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, bottom_up: torch.Tensor, top_down: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect spatial patterns and predict their evolution.
        
        Pattern types encoded:
            - 0-3: Symmetry types (horizontal, vertical, diagonal, rotational)
            - 4-7: Shape properties (size, position, orientation, color)
            - 8-11: Transformation patterns (translation, rotation, scaling, reflection)
            - 12-15: Relational patterns (inside, outside, above, below)
        """
        # Detect patterns from lower level
        features = self.pattern_detector(bottom_up)
        
        # Match against prototypes (soft clustering)
        similarities = torch.matmul(features, self.pattern_prototypes.T)
        pattern_activations = torch.softmax(similarities, dim=-1)
        
        # Combine with top-down if available
        if top_down is not None:
            pattern_activations = pattern_activations + 0.2 * top_down
            pattern_activations = torch.softmax(pattern_activations, dim=-1)
        
        # Generate pattern prediction
        weighted_prototypes = torch.matmul(pattern_activations, self.pattern_prototypes)
        prediction = self.pattern_predictor(weighted_prototypes)
        
        # Prediction error at pattern level
        prediction_error = torch.abs(bottom_up - prediction)
        
        return prediction, prediction_error
    
    def update(self, prediction_error: torch.Tensor, lr: float = 0.001):
        """Update pattern prototypes and predictor."""
        loss = prediction_error.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def extract_spatial_patterns(self, grid: np.ndarray) -> Dict[str, Any]:
        """
        Extract interpretable spatial patterns from grid.
        
        Returns dict with detected patterns:
            - symmetry: type and strength
            - shapes: detected shapes and properties
            - transformations: detected transformations
        """
        patterns = {
            'symmetry': self._detect_symmetry(grid),
            'shapes': self._detect_shapes(grid),
            'transformations': self._detect_transformations(grid),
        }
        return patterns
    
    def _detect_symmetry(self, grid: np.ndarray) -> Dict:
        """Detect symmetry patterns in grid."""
        h_sym = np.all(grid == np.fliplr(grid))
        v_sym = np.all(grid == np.flipud(grid))
        
        # Rotational symmetry
        rot90 = np.rot90(grid)
        rot_sym = np.all(grid == rot90)
        
        return {
            'horizontal': h_sym,
            'vertical': v_sym,
            'rotational': rot_sym,
            'strength': sum([h_sym, v_sym, rot_sym]) / 3.0
        }
    
    def _detect_shapes(self, grid: np.ndarray) -> List[Dict]:
        """Detect distinct shapes/connected components."""
        from scipy import ndimage
        
        shapes = []
        unique_colors = np.unique(grid[grid > 0])
        
        for color in unique_colors:
            mask = (grid == color).astype(int)
            labeled, num_features = ndimage.label(mask)
            
            for i in range(1, num_features + 1):
                coords = np.argwhere(labeled == i)
                if len(coords) > 0:
                    shapes.append({
                        'color': int(color),
                        'size': len(coords),
                        'centroid': coords.mean(axis=0).tolist(),
                        'bbox': [
                            int(coords[:, 0].min()),
                            int(coords[:, 1].min()),
                            int(coords[:, 0].max()),
                            int(coords[:, 1].max())
                        ]
                    })
        
        return shapes
    
    def _detect_transformations(self, grid: np.ndarray) -> Dict:
        """Detect likely transformations based on patterns."""
        # This would analyze sequential frames to infer transformations
        # For now, return empty structure
        return {
            'translation': None,
            'rotation': None,
            'reflection': None,
            'scaling': None,
        }


class Level3PredictiveLayer(PredictiveLayer):
    """
    Level 3: Structural/Goal prediction.
    Predicts high-level structure, task goals, and environmental dynamics.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__(input_dim, hidden_dim, input_dim)
        
        # Structural reasoning network
        self.structure_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Goal/Objective prediction
        self.goal_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64),  # Goal embedding
        )
        
        # Dynamics prediction (what happens next)
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        
    def forward(self, bottom_up: torch.Tensor, top_down: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Infer structural beliefs and predict goals/dynamics.
        
        Returns:
            prediction: Predicted structural state
            prediction_error: Error signal
        """
        # Extract structural features
        structure = self.structure_net(bottom_up)
        
        # Predict goal/objective embedding
        goal_embedding = self.goal_predictor(structure)
        
        # Predict next structural state (dynamics)
        next_structure = self.dynamics_predictor(structure)
        
        # Incorporate top-down priors if available
        if top_down is not None:
            next_structure = next_structure + 0.3 * top_down
        
        prediction_error = torch.abs(bottom_up - next_structure)
        
        return next_structure, prediction_error
    
    def update(self, prediction_error: torch.Tensor, lr: float = 0.0005):
        """Update structural model."""
        loss = prediction_error.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def infer_goal(self, features: torch.Tensor) -> str:
        """
        Infer likely task goal from structural features.
        
        Returns goal type:
            - 'reconstruct': Reconstruct target pattern
            - 'transform': Apply transformation
            - 'select': Select specific elements
            - 'sort': Organize elements
            - 'count': Numerical reasoning
        """
        with torch.no_grad():
            goal_emb = self.goal_predictor(features)
            # Map embedding to goal type (simplified)
            goal_idx = torch.argmax(goal_emb[:5]).item()
            goals = ['reconstruct', 'transform', 'select', 'sort', 'count']
            return goals[min(goal_idx, len(goals)-1)]


class PredictivePerception:
    """
    Hierarchical Predictive Perception system.
    
    Implements 3-level predictive coding hierarchy:
        Level 1: Pixel/color predictions
        Level 2: Spatial pattern predictions  
        Level 3: Structural/goal predictions
    
    Each level generates predictions that minimize Free Energy (prediction error).
    """
    
    def __init__(self, config: PerceptionConfig = None, grid_size: int = 64, num_colors: int = 16):
        self.config = config or PerceptionConfig()
        self.grid_size = grid_size
        self.num_colors = num_colors
        
        # Compute input dimension
        self.input_dim = grid_size * grid_size
        
        # Initialize hierarchical layers
        self.level1 = Level1PredictiveLayer(
            grid_size=grid_size,
            num_colors=num_colors,
            hidden_dim=self.config.perception.level1_hidden_dim
        )
        
        self.level2 = Level2PredictiveLayer(
            input_dim=self.config.perception.level1_hidden_dim // 2,
            hidden_dim=self.config.perception.level2_hidden_dim
        )
        
        self.level3 = Level3PredictiveLayer(
            input_dim=self.config.perception.level2_hidden_dim // 2,
            hidden_dim=self.config.perception.level3_hidden_dim
        )
        
        # History for learning
        self.prediction_history: List[Dict] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.level1.to(self.device)
        self.level2.to(self.device)
        self.level3.to(self.device)
        
    def infer(self, frame: np.ndarray) -> BeliefState:
        """
        Perform hierarchical inference on input frame.
        
        Implements predictive coding:
            1. Bottom-up: Extract features from observation
            2. Top-down: Generate predictions at each level
            3. Calculate prediction errors (Free Energy)
            
        Args:
            frame: Input grid [grid_size, grid_size] with values in [0, num_colors-1]
            
        Returns:
            BeliefState: Hierarchical beliefs and prediction errors
        """
        # Normalize input
        if frame.max() > 1:
            frame_norm = frame / (self.num_colors - 1)
        else:
            frame_norm = frame
        
        # Resize to expected grid size if necessary (using scipy for efficiency)
        if frame_norm.shape != (self.grid_size, self.grid_size):
            from scipy.ndimage import zoom
            zoom_y = self.grid_size / frame_norm.shape[0]
            zoom_x = self.grid_size / frame_norm.shape[1]
            frame_norm = zoom(frame_norm, (zoom_y, zoom_x), order=0)
            
        # Quantize colors for 1000 FPS optimization: 16 cores -> 4 cores (2 bits per pixel)
        # Reduces memory and speeds up processing
        if self.num_colors == 16:
            frame_quantized = (frame_norm * 15).astype(np.int32)  # Scale to 0-15
            frame_quantized = (frame_quantized // 4).astype(np.uint8)  # Map to 0-3
            frame_norm = frame_quantized / 3.0  # Normalize back to 0-1
            
        # Convert to tensor
        frame_flat = frame_norm.flatten()
        frame_tensor = torch.FloatTensor(frame_flat).unsqueeze(0).to(self.device)
        
        # Level 1: Pixel prediction
        pred1, err1 = self.level1(frame_tensor, None)
        
        # Level 2: Pattern prediction (uses encoded level 1 features)
        level1_encoded_raw = self.level1.encoder(frame_tensor)
        # Apply projection to ensure correct dimensions
        level1_encoded = self.level1.encoder_projection(level1_encoded_raw)
            
        pred2, err2 = self.level2(level1_encoded, None)
        
        # Level 3: Structural prediction (uses level2 output, not raw encoded)
        # Use pred2 which has correct dimensions for level3 input
        level2_for_l3 = pred2 if pred2.shape[1] == self.level3.input_dim else \
                       torch.nn.functional.adaptive_avg_pool1d(
                           pred2.unsqueeze(0), self.level3.input_dim
                       ).squeeze(0)
        pred3, err3 = self.level3(level2_for_l3, None)
        
        # Calculate total Free Energy
        free_energies = [
            err1.mean().item(),
            err2.mean().item(),
            err3.mean().item()
        ]
        
        # Extract interpretable features
        with torch.no_grad():
            patterns = self.level2.extract_spatial_patterns(frame)
            # Ensure correct dimensions for infer_goal
            l3_features = self.level3.structure_net(level2_for_l3.unsqueeze(0) if level2_for_l3.dim() == 1 else level2_for_l3)
            if l3_features.dim() > 1:
                l3_features = l3_features.squeeze(0)
            inferred_goal = self.level3.infer_goal(l3_features)
        
        # Convert to numpy
        beliefs = BeliefState(
            level1_features=pred1.detach().cpu().numpy().flatten(),
            level2_patterns=pred2.detach().cpu().numpy().flatten(),
            level3_structure=pred3.detach().cpu().numpy().flatten(),
            prediction_errors=free_energies,
            confidence=1.0 / (1.0 + sum(free_energies))
        )
        
        # Store for learning
        self.prediction_history.append({
            'frame': frame.copy(),
            'beliefs': beliefs,
            'patterns': patterns,
            'goal': inferred_goal,
            'free_energy': sum(free_energies)
        })
        
        return beliefs
    
    def learn(self, actual_next_frame: np.ndarray, lr: float = None):
        """
        Update predictions based on actual outcome.
        
        This implements the core predictive coding learning rule:
        minimize prediction error through weight updates.
        """
        if lr is None:
            lr = self.config.learning_rate
            
        if len(self.prediction_history) == 0:
            return
            
        # Get last prediction
        last = self.prediction_history[-1]
        
        # Normalize actual
        if actual_next_frame.max() > 1:
            actual_norm = actual_next_frame / (self.num_colors - 1)
        else:
            actual_norm = actual_next_frame
            
        actual_tensor = torch.FloatTensor(actual_norm.flatten()).unsqueeze(0).to(self.device)
        
        # Calculate prediction errors and update each level
        with torch.no_grad():
            pred1 = torch.FloatTensor(last['beliefs'].level1_features).unsqueeze(0).to(self.device)
            err1 = torch.abs(actual_tensor - pred1)
            
        self.level1.update(err1, lr)
        
        # Propagate errors up the hierarchy
        # (In full implementation, would update level 2 and 3 as well)
        
    def get_free_energy(self) -> float:
        """Get total Free Energy of current beliefs."""
        if len(self.prediction_history) == 0:
            return float('inf')
        return self.prediction_history[-1]['free_energy']
    
    def reset(self):
        """Reset perception state."""
        self.prediction_history.clear()
