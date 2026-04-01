"""Layer 2: Active Inference Engine - Free Energy Minimization.

Implements Karl Friston's Active Inference framework based on the Free Energy Principle.
The agent selects actions that minimize expected Free Energy (surprise), rather than
maximizing external reward.

Key insight: Perception and action are both inference processes that minimize Free Energy:
    - Perception: infer hidden states that minimize Free Energy (sensory surprise)
    - Action: select policies that minimize expected Free Energy (future surprise)

References:
    - Friston, K. et al. (2017). Active inference: a process theory.
    - Friston, K. (2019). A free energy principle for a particular physics.
    - Parr et al. (2022). "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior"
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import numpy as np
from scipy.stats import entropy
import torch
import torch.nn.functional as F
from torch.distributions import Distribution

from .config import ActiveInferenceConfig
from .perception import BeliefState


class ActionType(Enum):
    """ARC-AGI-3 action types."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"
    UNDO = "undo"
    CLICK = "click"


@dataclass
class Policy:
    """A sequence of actions (policy) for planning."""
    actions: List[str]
    expected_free_energy: float = float('inf')
    posterior_probability: float = 0.0


@dataclass
class GenerativeModel:
    """
    The agent's internal model of the environment.
    
    In Active Inference, the agent has a generative model:
        p(o, s, π) = p(π) * p(s₁|π) * ∏ p(oₜ|sₜ) * p(sₜ₊₁|sₜ, π)
    
    Where:
        - o: observations
        - s: hidden states
        - π: policies (action sequences)
    """
    
    def __init__(self, state_dim: int, action_dim: int, obs_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        # Transition model: p(s' | s, a)
        # Approximated as neural network
        self.transition_net = self._build_transition_net()
        
        # Observation model: p(o | s)
        self.observation_net = self._build_observation_net()
        
        # Prior preferences over observations (what the agent "wants")
        self.preference_prior = np.zeros(obs_dim)
        
    def _build_transition_net(self):
        """Build network for p(s' | s, a)."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.state_dim)
        )
    
    def _build_observation_net(self):
        """Build network for p(o | s)."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.obs_dim)
        )
    
    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Predict next state given current state and action.
        p(s' | s, a)
        """
        s = torch.FloatTensor(state)
        a = torch.zeros(self.action_dim)
        a[action] = 1.0
        
        inp = torch.cat([s, a])
        with torch.no_grad():
            next_s = self.transition_net(inp.unsqueeze(0))
        return next_s.squeeze(0).numpy()
    
    def observation_likelihood(self, state: np.ndarray) -> np.ndarray:
        """
        Predict observation from state.
        p(o | s)
        """
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            obs = self.observation_net(s)
        return torch.sigmoid(obs).squeeze(0).numpy()
    
    def variational_free_energy(self, q_dist: Distribution, p_prior: Distribution, 
                                 likelihood: float) -> torch.Tensor:
        """
        F = E_q[ln q(s) - ln p(s,o)] = KL[q||p] - ln p(o)
        Minimizar F ≈ minimizar surpresa + manter crenças simples
        
        Implementation based on Parr et al. (2022)
        """
        kl_divergence = torch.distributions.kl.kl_divergence(q_dist, p_prior)
        surprise = -torch.log(torch.tensor(likelihood) + 1e-8)
        return kl_divergence + surprise
    
    def _state_to_hash(self, state: np.ndarray, action: str) -> str:
        """Convert state-action pair to hash for caching."""
        state_bytes = state.tobytes()
        return hashlib.md5(state_bytes + action.encode()).hexdigest()[:16]


class ActiveInferenceAgent:
    """
    Active Inference Agent implementing Free Energy minimization.
    
    The agent follows the principle that both perception and action aim to minimize
    variational Free Energy, which bounds sensory surprise.
    
    Free Energy decomposes as:
        F = E_q[ln q(s) - ln p(o, s)] = D_KL[q(s) || p(s|o)] - ln p(o)
          = Complexity - Accuracy
          = D_KL[q(s) || p(s)] - E_q[ln p(o|s)]
          
    For action selection:
        G(π) = E_{q(o,s|π)}[ln q(s|π) - ln p(o,s)] = Expected Free Energy
             = Expected complexity - Expected accuracy
             = D_KL[q(o|π) || p(o)] - E_q[ln p(o|s)]
    
    The agent selects the policy π that minimizes expected Free Energy G(π).
    """
    
    def __init__(self, config: ActiveInferenceConfig = None, 
                 grid_size: int = 64, num_colors: int = 16):
        self.config = config or ActiveInferenceConfig()
        self.grid_size = grid_size
        self.num_colors = num_colors
        
        # Action space (ARC-AGI-3)
        self.actions = ["up", "down", "left", "right", "stay", "undo"]
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
        # Generative model
        obs_dim = grid_size * grid_size
        state_dim = 512  # Latent state dimension
        self.model = GenerativeModel(
            state_dim=state_dim,
            action_dim=len(self.actions),
            obs_dim=obs_dim
        )
        
        # Current beliefs about hidden states
        self.current_beliefs: Optional[np.ndarray] = None
        
        # Planning cache
        self.policy_cache: List[Policy] = []
        self.history: List[Dict] = []
        
        # Learning
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(
            list(self.model.transition_net.parameters()) + 
            list(self.model.observation_net.parameters()),
            lr=self.learning_rate
        )
        
    def free_energy(self, observation: np.ndarray, prediction: np.ndarray,
                   prior: np.ndarray) -> float:
        """
        Calculate variational Free Energy.
        
        F = E_q[ln q(s) - ln p(o, s)]
          = D_KL[q(s) || p(s)] - E_q[ln p(o|s)]
          = Complexity - Accuracy
        
        Args:
            observation: Actual observation
            prediction: Predicted observation
            prior: Prior beliefs
            
        Returns:
            Free Energy (scalar)
        """
        # Ensure arrays
        obs = np.asarray(observation).flatten()
        pred = np.asarray(prediction).flatten()
        
        # Accuracy term: -E_q[ln p(o|s)] ≈ -log p(o|s) for Gaussian
        # Using squared error as proxy
        accuracy = -0.5 * np.sum((obs - pred) ** 2)
        
        # Complexity term: D_KL[q(s) || p(s)]
        # Approximate using entropy difference
        prior_entropy = entropy(np.abs(prior) + 1e-10)
        post_entropy = entropy(np.abs(pred) + 1e-10)
        complexity = np.abs(post_entropy - prior_entropy)
        
        # Free Energy = Complexity - Accuracy
        free_energy = complexity - accuracy * 0.1  # Scale accuracy
        
        return float(free_energy)
    
    @lru_cache(maxsize=10000)
    def cached_predict(self, state_hash: str, action: str) -> np.ndarray:
        """
        Cached prediction for state-action pair.
        Uses LRU cache for 1000 FPS performance optimization.
        """
        # Note: state_hash is a string representation for cacheability
        # The actual prediction uses the current model state
        with torch.no_grad():
            s = torch.zeros(self.model.state_dim)
            a_idx = self.action_to_idx.get(action, 0)
            a = torch.zeros(self.model.action_dim)
            a[a_idx] = 1.0
            
            inp = torch.cat([s, a])
            next_s = self.model.transition_net(inp.unsqueeze(0))
            return next_s.squeeze(0).numpy()
    
    def expected_free_energy(self, policy: Policy, current_beliefs: np.ndarray,
                            horizon: int = None) -> float:
        """
        Calculate expected Free Energy for a policy (action sequence).
        
        G(π) = Σ_τ G(π, τ)
        
        Where G(π, τ) decomposes as:
            - Information gain (expected reduction in uncertainty)
            - Expected preferences (how much agent "wants" predicted observations)
        
        Args:
            policy: Sequence of actions
            current_beliefs: Current belief state
            horizon: Planning horizon
            
        Returns:
            Expected Free Energy for this policy
        """
        if horizon is None:
            horizon = self.config.horizon
            
        G = 0.0
        simulated_beliefs = current_beliefs.copy()
        
        for t, action in enumerate(policy.actions[:horizon]):
            # Simulate transition
            action_idx = self.action_to_idx.get(action, 0)
            next_beliefs = self.model.transition(simulated_beliefs, action_idx)
            
            # Predict observation
            predicted_obs = self.model.observation_likelihood(next_beliefs)
            
            # Calculate Free Energy at this timestep
            # Components:
            # 1. Ambiguity (predictive uncertainty)
            ambiguity = -np.var(predicted_obs)
            
            # 2. Risk (KL between predicted and preferred observations)
            risk = np.sum(
                predicted_obs * np.log(predicted_obs / (self.model.preference_prior + 1e-10) + 1e-10)
            )
            
            # 3. Information gain (expected Bayesian surprise)
            info_gain = np.abs(np.var(next_beliefs) - np.var(simulated_beliefs))
            
            # Combine components (weighted sum)
            timestep_fe = (
                self.config.complexity_weight * risk +
                ambiguity * 0.1 +
                info_gain * 0.5
            )
            
            G += (self.config.gamma ** t) * timestep_fe
            
            simulated_beliefs = next_beliefs
            
        return G
    
    def select_action(self, beliefs: BeliefState, possible_actions: List[str],
                     num_samples: int = None) -> Tuple[str, Dict]:
        """
        Select action by minimizing expected Free Energy.
        
        This is the core Active Inference action selection:
            1. Generate candidate policies (action sequences)
            2. Calculate expected Free Energy for each
            3. Select policy with lowest G(π)
            4. Execute first action of winning policy
        
        Args:
            beliefs: Current belief state from perception
            possible_actions: Available actions
            num_samples: Number of policy samples to evaluate
            
        Returns:
            selected_action: The action to take
            info: Additional information (Free Energy values, etc.)
        """
        if num_samples is None:
            num_samples = self.config.num_samples
            
        # Convert beliefs to state vector
        if self.current_beliefs is None:
            # Ensure exactly state_dim (512) dimensions
            feats = np.concatenate([
                beliefs.level1_features[:170],
                beliefs.level2_patterns[:170],
                beliefs.level3_structure[:172]
            ])
            # Pad or truncate to exactly 512
            if len(feats) < 512:
                feats = np.pad(feats, (0, 512 - len(feats)))
            elif len(feats) > 512:
                feats = feats[:512]
            self.current_beliefs = feats
        
        # Generate candidate policies
        policies = self._generate_policies(possible_actions, num_samples)
        
        # Evaluate each policy
        policy_scores = []
        for policy in policies:
            g = self.expected_free_energy(policy, self.current_beliefs)
            policy.expected_free_energy = g
            policy_scores.append((policy, g))
        
        # Convert to posterior probabilities (softmax over negative FE)
        scores = np.array([s for _, s in policy_scores])
        
        # Add temperature for exploration
        temp = self.config.temperature
        log_probs = -scores / temp
        log_probs = log_probs - np.max(log_probs)  # Numerical stability
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)
        
        for i, (policy, _) in enumerate(policy_scores):
            policy.posterior_probability = probs[i]
        
        # Select best policy (argmin of expected Free Energy)
        best_policy, best_g = min(policy_scores, key=lambda x: x[1])
        
        # Get first action from best policy
        selected_action = best_policy.actions[0]
        
        info = {
            'expected_free_energy': best_g,
            'policy_probability': best_policy.posterior_probability,
            'all_policies': [
                {'actions': p.actions, 'fe': p.expected_free_energy, 'prob': p.posterior_probability}
                for p, _ in policy_scores
            ],
            'beliefs_confidence': beliefs.confidence
        }
        
        # Store history
        self.history.append({
            'beliefs': beliefs,
            'selected_action': selected_action,
            'free_energy': best_g,
            'policy': best_policy
        })
        
        return selected_action, info
    
    def _generate_policies(self, possible_actions: List[str], 
                          num_samples: int) -> List[Policy]:
        """
        Generate candidate action sequences (policies).
        
        For computational efficiency, we sample policies rather than enumerating
        all possibilities (exponential in horizon).
        """
        policies = []
        horizon = min(self.config.horizon, 3)  # Limit horizon for efficiency
        
        # Always include single-action policies
        for action in possible_actions:
            policies.append(Policy(actions=[action]))
        
        # Sample multi-action policies
        for _ in range(num_samples - len(possible_actions)):
            actions = []
            for _ in range(horizon):
                action = np.random.choice(possible_actions)
                actions.append(action)
            policies.append(Policy(actions=actions))
        
        return policies
    
    def simulate(self, beliefs: BeliefState, action: str) -> Tuple[np.ndarray, float]:
        """
        Simulate the consequence of taking an action.
        
        Returns:
            predicted_frame: Predicted next observation
            confidence: Confidence in prediction (inverse of expected uncertainty)
        """
        if self.current_beliefs is None:
            self.current_beliefs = np.concatenate([
                beliefs.level1_features,
                beliefs.level2_patterns[:256],
                beliefs.level3_structure[:256]
            ])[:512]
        
        # Execute transition
        action_idx = self.action_to_idx.get(action, 0)
        next_beliefs = self.model.transition(self.current_beliefs, action_idx)
        
        # Generate predicted observation
        predicted_flat = self.model.observation_likelihood(next_beliefs)
        
        # Reshape to grid
        predicted_frame = predicted_flat.reshape(self.grid_size, self.grid_size)
        
        # Confidence based on prediction variance
        confidence = 1.0 / (1.0 + np.var(predicted_flat))
        
        return predicted_frame, confidence
    
    def update_beliefs(self, state: np.ndarray, action: str, 
                      next_state: np.ndarray, reward: float = None):
        """
        Update the generative model based on observed transition.
        
        This implements learning in Active Inference: updating the model
        to better predict observations (minimize Free Energy).
        """
        # Prepare tensors - ensure correct dimensions
        s = torch.FloatTensor(state.flatten()[:self.model.state_dim]).unsqueeze(0)
        if s.shape[1] < self.model.state_dim:
            # Pad if necessary
            padding = torch.zeros(1, self.model.state_dim - s.shape[1])
            s = torch.cat([s, padding], dim=1)
        
        s_next = torch.FloatTensor(next_state.flatten()[:self.model.state_dim]).unsqueeze(0)
        if s_next.shape[1] < self.model.state_dim:
            padding = torch.zeros(1, self.model.state_dim - s_next.shape[1])
            s_next = torch.cat([s_next, padding], dim=1)
        
        a_idx = self.action_to_idx.get(action, 0)
        a = torch.zeros(1, len(self.actions))
        a[0, a_idx] = 1.0
        
        # Training step: minimize prediction error
        inp = torch.cat([s, a], dim=1)
        pred_s_next = self.model.transition_net(inp)
        
        # Loss: reconstruction error
        loss = F.mse_loss(pred_s_next, s_next)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update current beliefs
        with torch.no_grad():
            self.current_beliefs = pred_s_next.squeeze(0).numpy()
    
    def set_preferences(self, preferred_observations: np.ndarray):
        """
        Set prior preferences over observations.
        
        This encodes what the agent "wants" to observe, shaping its behavior
        through the risk term in expected Free Energy.
        """
        self.model.preference_prior = preferred_observations.flatten()
    
    def reset(self):
        """Reset agent state."""
        self.current_beliefs = None
        self.policy_cache.clear()
        self.history.clear()
