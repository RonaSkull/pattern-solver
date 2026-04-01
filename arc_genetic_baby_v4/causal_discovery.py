"""
Causal Discovery Engine for ARC-AGI-3
Implements Pearl's Do-Calculus + PC Algorithm for learning causal structures
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from scipy import stats
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class CausalMechanism(Enum):
    """Types of causal mechanisms"""
    DIRECT = auto()
    MEDIATED = auto()
    CONFOUNDED = auto()
    SPURIOUS = auto()


@dataclass
class CausalVariable:
    """Represents a variable in the causal domain"""
    name: str
    domain: np.ndarray
    is_observed: bool = True
    is_intervenable: bool = True


@dataclass
class CausalHypothesis:
    """Causal hypothesis with Bayesian confidence"""
    cause: str
    effect: str
    mechanism: CausalMechanism
    strength: float = 0.0
    confidence: float = 0.5
    evidence_count: int = 0
    
    def update(self, success: bool, alpha: float = 0.1):
        """Bayesian update of confidence"""
        self.evidence_count += 1
        self.confidence = (1 - alpha) * self.confidence + alpha * float(success)
        self.strength = min(1.0, self.strength + 0.01 * float(success))


class CausalGraph:
    """Dynamic causal graph with structural learning"""
    
    def __init__(self, variables: List[CausalVariable], max_parents: int = 3):
        self.graph = nx.DiGraph()
        self.variables = {v.name: v for v in variables}
        self.hypotheses: Dict[Tuple[str, str], CausalHypothesis] = {}
        self.max_parents = max_parents
        
        for var in variables:
            self.graph.add_node(var.name, variable=var)
    
    def add_edge(self, cause: str, effect: str, 
                 mechanism: CausalMechanism = CausalMechanism.DIRECT,
                 strength: float = 0.1) -> bool:
        """Add causal edge to graph"""
        if cause not in self.variables or effect not in self.variables:
            return False
        if self.graph.has_edge(cause, effect):
            return False
        if self.graph.in_degree(effect) >= self.max_parents:
            return False
        
        self.graph.add_edge(cause, effect, mechanism=mechanism)
        
        key = (cause, effect)
        if key not in self.hypotheses:
            self.hypotheses[key] = CausalHypothesis(
                cause=cause, effect=effect, mechanism=mechanism,
                strength=strength, confidence=0.5
            )
        return True
    
    def do_intervention(self, variable: str, value: Any) -> Dict[str, Any]:
        """Simulate causal intervention P(Y | do(X=x))"""
        if variable not in self.graph:
            return {}
        
        descendants = nx.descendants(self.graph, variable)
        effects = {variable: value}
        
        for node in descendants:
            parents = list(self.graph.predecessors(node))
            if variable in parents:
                key = (variable, node)
                if key in self.hypotheses:
                    hyp = self.hypotheses[key]
                    if hyp.confidence > 0.3:
                        effects[node] = self._propagate_effect(value, hyp, node)
        
        return effects
    
    def _propagate_effect(self, parent_value: Any, 
                         hypothesis: CausalHypothesis, 
                         target: str) -> Any:
        """Propagate causal effect through network"""
        noise = np.random.normal(0, 0.1 * (1 - hypothesis.confidence))
        result = parent_value + hypothesis.strength * noise
        return int(np.clip(result, 0, 15))
    
    def learn_structure(self, observations: List[Dict], 
                       max_iterations: int = 50) -> float:
        """Learn causal structure from observations"""
        if len(observations) < 10:
            return 0.0
        
        var_names = list(self.variables.keys())
        
        # Test dependencies and add edges
        for cause, effect in [(c, e) for c in var_names for e in var_names if c != e]:
            if self._test_dependency(observations, cause, effect):
                self.add_edge(cause, effect)
        
        # Prune weak edges
        for (cause, effect), hyp in list(self.hypotheses.items()):
            if not hyp.is_valid() if hasattr(hyp, 'is_valid') else hyp.confidence < 0.3:
                if self.graph.has_edge(cause, effect):
                    self.graph.remove_edge(cause, effect)
        
        return len(self.hypotheses)
    
    def _test_dependency(self, observations: List[Dict], 
                        var1: str, var2: str) -> bool:
        """Test statistical dependency between variables"""
        vals1 = [obs.get(var1) for obs in observations if var1 in obs and var2 in obs]
        vals2 = [obs.get(var2) for obs in observations if var1 in obs and var2 in obs]
        
        if len(vals1) < 5:
            return False
        
        try:
            corr, p_value = stats.pearsonr(vals1, vals2)
            return abs(corr) > 0.3 and p_value < 0.1
        except:
            return False
    
    def get_causal_parents(self, variable: str) -> List[str]:
        """Get direct causes of variable"""
        return list(self.graph.predecessors(variable))
    
    def get_all_causes(self, variable: str) -> Set[str]:
        """Get all ancestors (causes) of variable"""
        return nx.ancestors(self.graph, variable)


class CausalDiscoveryEngine:
    """
    Main causal discovery engine for ARC-AGI-3
    Integrates observation, intervention, and structural learning
    """
    
    def __init__(self, grid_size: int = 64, num_colors: int = 16):
        self.grid_size = grid_size
        self.num_colors = num_colors
        
        # Define ARC-specific causal variables
        self.variables = self._define_variables()
        self.causal_graph = CausalGraph(self.variables)
        
        # Buffers for learning
        self.observation_buffer: List[Dict] = []
        self.intervention_history: List[Dict] = []
        self.current_state: Optional[Dict] = None
        
        self.step_count = 0
        self.learn_interval = 10
    
    def _define_variables(self) -> List[CausalVariable]:
        """Define causal variables for ARC domain"""
        return [
            CausalVariable("object_position_x", domain=np.arange(self.grid_size)),
            CausalVariable("object_position_y", domain=np.arange(self.grid_size)),
            CausalVariable("object_size", domain=np.arange(1, self.grid_size//2)),
            CausalVariable("object_color", domain=np.arange(self.num_colors)),
            CausalVariable("background_color", domain=np.arange(self.num_colors)),
            CausalVariable("transformation_type", domain=np.arange(6), is_intervenable=True),
            CausalVariable("output_pattern", domain=np.arange(5)),
        ]
    
    def observe(self, grid: np.ndarray, context: Dict = None) -> Dict[str, Any]:
        """Extract causal features from grid"""
        obs = {}
        
        # Extract objects
        objects = self._extract_objects(grid)
        if objects:
            main_obj = max(objects, key=lambda o: len(o['pixels']))
            obs['object_position_x'] = main_obj['centroid'][0]
            obs['object_position_y'] = main_obj['centroid'][1]
            obs['object_size'] = len(main_obj['pixels'])
            obs['object_color'] = main_obj['color']
        
        obs['background_color'] = self._estimate_background(grid)
        
        if context:
            obs.update(context)
        
        self.observation_buffer.append(obs)
        self.current_state = obs
        
        return obs
    
    def intervene(self, action: str, param: int = None, 
                 context: Dict = None) -> Dict[str, Any]:
        """Execute causal intervention"""
        action_map = {
            'rotate': ('transformation_type', 0),
            'flip_h': ('transformation_type', 1),
            'flip_v': ('transformation_type', 2),
            'color_shift': ('transformation_type', 3),
            'translate': ('transformation_type', 4),
            'stay': ('transformation_type', 5),
        }
        
        if action not in action_map:
            return {}
        
        var_name, default_param = action_map[action]
        param = param if param is not None else default_param
        
        intervention = {
            'variable': var_name,
            'value': param,
            'pre_state': self.current_state.copy() if self.current_state else {},
            'causal_var': var_name,
            'timestamp': self.step_count
        }
        
        expected_effects = self.causal_graph.do_intervention(var_name, param)
        intervention['expected_effects'] = expected_effects
        self.intervention_history.append(intervention)
        
        self.step_count += 1
        
        # Periodic learning
        if self.step_count % self.learn_interval == 0:
            self._trigger_learning()
        
        return expected_effects
    
    def learn_from_outcome(self, outcome: Dict, reward: float = None) -> float:
        """Update causal model from outcome"""
        if not self.intervention_history:
            return 0.0
        
        last = self.intervention_history[-1]
        expected = last.get('expected_effects', {})
        
        # Update edge strengths
        for var_name, expected_val in expected.items():
            actual_val = outcome.get(var_name)
            if actual_val is not None:
                key = (last['causal_var'], var_name)
                if key in self.causal_graph.hypotheses:
                    success = abs(expected_val - actual_val) < 2
                    self.causal_graph.hypotheses[key].update(success)
        
        return 0.1  # Learning signal
    
    def query_causal_effects(self, action: str, 
                            context: Dict = None) -> List[Tuple[str, float, float]]:
        """Query expected effects of action"""
        expected = self.intervene(action, context=context)
        
        results = []
        if self.intervention_history:
            for var_name in expected:
                key = (self.intervention_history[-1]['causal_var'], var_name)
                if key in self.causal_graph.hypotheses:
                    hyp = self.causal_graph.hypotheses[key]
                    results.append((var_name, hyp.strength, hyp.confidence))
        
        return sorted(results, key=lambda x: x[1] * x[2], reverse=True)
    
    def _trigger_learning(self):
        """Trigger structural learning"""
        if len(self.observation_buffer) >= 20:
            sample = self.observation_buffer[-50:]
            self.causal_graph.learn_structure(sample)
            self.observation_buffer = self.observation_buffer[-25:]
    
    def _extract_objects(self, grid: np.ndarray) -> List[Dict]:
        """Extract connected objects from grid"""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
        
        for color in range(self.num_colors):
            mask = (grid == color) & ~visited
            if not np.any(mask):
                continue
            
            labeled, num = ndimage.label(mask.astype(int), structure=structure)
            
            for obj_id in range(1, num + 1):
                obj_mask = (labeled == obj_id)
                pixels = list(zip(*np.where(obj_mask)))
                
                if len(pixels) >= 2:
                    centroid = np.mean(pixels, axis=0).astype(int)
                    objects.append({
                        'color': color,
                        'pixels': pixels,
                        'centroid': tuple(centroid),
                    })
                    visited[obj_mask] = True
        
        return objects
    
    def _estimate_background(self, grid: np.ndarray) -> int:
        """Estimate background color from borders"""
        border = np.concatenate([grid[0,:], grid[-1,:], grid[:,0], grid[:,-1]])
        return int(np.bincount(border).argmax())
    
    def reset(self):
        """Reset for new episode"""
        self.current_state = None
    
    def save_checkpoint(self, path: str):
        """Save causal engine state"""
        import json
        from pathlib import Path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'variables': [{'name': v.name, 'domain': v.domain.tolist()} 
                         for v in self.variables],
            'hypotheses': [{
                'cause': h.cause,
                'effect': h.effect,
                'strength': h.strength,
                'confidence': h.confidence,
                'evidence': h.evidence_count
            } for h in self.causal_graph.hypotheses.values()],
            'step_count': self.step_count
        }
        
        with open(path / 'causal_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, path: str) -> 'CausalDiscoveryEngine':
        """Load causal engine from checkpoint"""
        import json
        from pathlib import Path
        
        with open(Path(path) / 'causal_checkpoint.json', 'r') as f:
            data = json.load(f)
        
        # Reconstruct
        first_var = data['variables'][0] if data['variables'] else None
        grid_size = max(first_var['domain']) + 1 if first_var else 10
        
        engine = cls(grid_size=grid_size)
        
        # Restore hypotheses
        for h_data in data['hypotheses']:
            engine.causal_graph.add_edge(
                h_data['cause'], h_data['effect'],
                strength=h_data['strength']
            )
            key = (h_data['cause'], h_data['effect'])
            if key in engine.causal_graph.hypotheses:
                hyp = engine.causal_graph.hypotheses[key]
                hyp.confidence = h_data['confidence']
                hyp.evidence_count = h_data['evidence']
        
        engine.step_count = data.get('step_count', 0)
        
        return engine
