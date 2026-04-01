"""
Counterfactual World Model for ARC-AGI-3
Simulates "what-if" scenarios with causal interventions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
import copy
from dataclasses import dataclass
from enum import Enum


class CounterfactualType(Enum):
    """Types of counterfactual queries"""
    ACTION_OUTCOME = "what_if_action"
    STATE_INTERVENTION = "what_if_state"
    PATH_ALTERNATIVE = "what_if_path"


@dataclass
class CounterfactualScenario:
    """A counterfactual scenario to simulate"""
    initial_state: np.ndarray
    intervention: Dict[str, Any]
    expected_outcome: Optional[np.ndarray] = None
    confidence: float = 0.0


class CounterfactualWorldModel(nn.Module):
    """
    Neural model for counterfactual simulation.
    Predicts outcomes given interventions.
    """
    
    def __init__(self, grid_size: int = 64, num_colors: int = 16, 
                 hidden_dim: int = 256):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_dim),
            nn.ReLU()
        )
        
        # Intervention encoder
        self.intervention_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_size * grid_size * num_colors)
        )
        
    def forward(self, state: torch.Tensor, 
                intervention: torch.Tensor) -> torch.Tensor:
        """
        Predict counterfactual outcome
        
        Args:
            state: [B, 1, H, W] grid state
            intervention: [B, 128] intervention encoding
            
        Returns:
            [B, num_colors, H, W] predicted outcome distribution
        """
        state_feat = self.state_encoder(state)
        int_feat = self.intervention_encoder(intervention)
        
        combined = torch.cat([state_feat, int_feat], dim=1)
        logits = self.predictor(combined)
        
        # Reshape to [B, num_colors, H, W]
        logits = logits.view(-1, self.num_colors, self.grid_size, self.grid_size)
        
        return torch.softmax(logits, dim=1)


class CounterfactualEngine:
    """
    Engine for counterfactual reasoning in ARC-AGI-3
    
    Capabilities:
    - Simulate action consequences
    - Compare alternative paths
    - Generate counterfactual explanations
    """
    
    def __init__(self, grid_size: int = 64, num_colors: int = 16,
                 device: str = 'cpu'):
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.device = device
        
        # Neural world model
        self.world_model = CounterfactualWorldModel(
            grid_size, num_colors
        ).to(device)
        
        # Simulation history for learning
        self.simulation_history: List[Tuple] = []
        
    def simulate_action(self, current_state: np.ndarray, 
                       action: str, 
                       num_samples: int = 10) -> List[np.ndarray]:
        """
        Simulate possible outcomes of taking an action
        
        Args:
            current_state: Current grid state [H, W]
            action: Action to simulate
            num_samples: Number of stochastic samples
            
        Returns:
            List of possible outcome grids
        """
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        
        # Encode intervention (action)
        intervention = self._encode_action(action)
        
        # Generate multiple samples
        outcomes = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred_dist = self.world_model(state_tensor, intervention)
                
                # Sample from distribution
                sampled = torch.multinomial(
                    pred_dist.view(-1, self.num_colors).transpose(0, 1),
                    1
                ).view(self.grid_size, self.grid_size)
                
                outcomes.append(sampled.cpu().numpy())
        
        return outcomes
    
    def compare_paths(self, state: np.ndarray,
                     action_a: str, 
                     action_b: str) -> Dict[str, Any]:
        """
        Compare two action sequences counterfactually
        
        Returns:
            Dictionary with comparison metrics
        """
        outcomes_a = self.simulate_action(state, action_a, num_samples=5)
        outcomes_b = self.simulate_action(state, action_b, num_samples=5)
        
        # Compute diversity within each action
        diversity_a = self._compute_diversity(outcomes_a)
        diversity_b = self._compute_diversity(outcomes_b)
        
        # Compute expected difference between actions
        diff_score = self._compute_path_difference(outcomes_a, outcomes_b)
        
        return {
            'action_a_diversity': diversity_a,
            'action_b_diversity': diversity_b,
            'expected_difference': diff_score,
            'recommended': action_a if diversity_a < diversity_b else action_b
        }
    
    def generate_explanation(self, state: np.ndarray,
                            action: str,
                            actual_outcome: np.ndarray) -> str:
        """
        Generate counterfactual explanation for an outcome
        
        Example: "If you hadn't moved left, the blue object 
                 would not have touched the border"
        """
        # Simulate alternative (no action)
        alt_outcomes = self.simulate_action(state, "stay", num_samples=3)
        alt_mean = np.mean([o == actual_outcome for o in alt_outcomes])
        
        # Compare with actual
        if alt_mean > 0.8:
            return f"Action '{action}' was critical - without it, outcome would differ significantly"
        else:
            return f"Action '{action}' may have been redundant - similar outcome likely without it"
    
    def learn_from_experience(self, state: np.ndarray,
                             action: str,
                             actual_outcome: np.ndarray):
        """Update world model from real experience"""
        self.simulation_history.append({
            'state': state.copy(),
            'action': action,
            'outcome': actual_outcome.copy()
        })
        
        # Trigger training if enough data
        if len(self.simulation_history) >= 32:
            self._train_step()
    
    def _encode_action(self, action: str) -> torch.Tensor:
        """Encode action string to tensor"""
        # Simple encoding - can be improved
        action_map = {
            'up': 0, 'down': 1, 'left': 2, 'right': 3,
            'stay': 4, 'undo': 5, 'click': 6
        }
        
        idx = action_map.get(action, 4)
        encoding = torch.zeros(1, 128)
        encoding[0, idx * 16:(idx + 1) * 16] = 1.0
        
        return encoding.to(self.device)
    
    def _compute_diversity(self, outcomes: List[np.ndarray]) -> float:
        """Compute diversity of outcomes"""
        if len(outcomes) < 2:
            return 0.0
        
        pairwise_diffs = []
        for i, o1 in enumerate(outcomes):
            for o2 in outcomes[i+1:]:
                diff = np.mean(o1 != o2)
                pairwise_diffs.append(diff)
        
        return np.mean(pairwise_diffs)
    
    def _compute_path_difference(self, outcomes_a: List[np.ndarray],
                                  outcomes_b: List[np.ndarray]) -> float:
        """Compute expected difference between two action paths"""
        diffs = []
        for oa in outcomes_a:
            for ob in outcomes_b:
                diffs.append(np.mean(oa != ob))
        return np.mean(diffs)
    
    def _train_step(self):
        """Train world model on accumulated experience"""
        # Simplified training - in production use proper DataLoader
        batch = self.simulation_history[-32:]
        
        states = torch.FloatTensor([e['state'] for e in batch])
        states = states.unsqueeze(1).to(self.device)
        
        actions = [e['action'] for e in batch]
        interventions = torch.cat([self._encode_action(a) for a in actions])
        
        outcomes = torch.LongTensor([e['outcome'] for e in batch])
        outcomes = outcomes.to(self.device)
        
        # Forward
        pred = self.world_model(states, interventions)
        
        # Loss: cross-entropy
        loss = nn.functional.cross_entropy(
            pred.view(-1, self.num_colors),
            outcomes.view(-1)
        )
        
        # Backward (simplified - in practice use optimizer)
        loss.backward()
        
        # Clear history
        self.simulation_history = []


class CounterfactualPlanner:
    """
    Planner that uses counterfactual simulation for decision making
    """
    
    def __init__(self, engine: CounterfactualEngine, horizon: int = 3):
        self.engine = engine
        self.horizon = horizon
        
    def plan(self, state: np.ndarray, 
            available_actions: List[str]) -> Tuple[str, float]:
        """
        Plan best action sequence using counterfactual simulation
        
        Returns:
            (best_action, expected_value)
        """
        action_scores = []
        
        for action in available_actions:
            # Simulate outcomes
            outcomes = self.engine.simulate_action(state, action, num_samples=5)
            
            # Score based on:
            # 1. Expected progress toward goal
            # 2. Low uncertainty (diversity)
            # 3. Novelty (not seen before)
            
            progress_scores = [self._score_progress(o) for o in outcomes]
            diversity = self.engine._compute_diversity(outcomes)
            
            # Higher progress, lower diversity = better
            score = np.mean(progress_scores) * (1 - diversity)
            action_scores.append((action, score))
        
        # Select best
        best_action, best_score = max(action_scores, key=lambda x: x[1])
        return best_action, best_score
    
    def _score_progress(self, outcome: np.ndarray) -> float:
        """Heuristic for progress toward goal"""
        # Simplified: reward grids with structure
        num_colors = len(np.unique(outcome))
        has_structure = num_colors > 1 and num_colors < 8
        
        return 1.0 if has_structure else 0.5
