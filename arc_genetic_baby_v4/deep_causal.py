"""
Deep Causal Reasoning Module for ARC-AGI-3
Implements 2nd+ order causality and latent variable discovery
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import itertools


class CausalOrder(Enum):
    """Order of causal relationship"""
    FIRST_ORDER = 1      # Direct cause-effect
    SECOND_ORDER = 2     # Cause of cause
    THIRD_ORDER = 3      # Meta-causal
    HIGHER_ORDER = 4     # Deep causality


@dataclass
class LatentVariable:
    """A hidden variable inferred from observations"""
    name: str
    inferred_from: List[str]  # Observable variables that imply this
    distribution: np.ndarray  # Estimated distribution
    confidence: float = 0.5
    causal_children: List[str] = field(default_factory=list)


@dataclass
class CausalMechanism:
    """Mechanism explaining how cause produces effect"""
    cause: str
    effect: str
    mechanism_type: str  # "direct", "mediated", "modulated", etc.
    mediators: List[str] = field(default_factory=list)
    moderators: Dict[str, float] = field(default_factory=dict)
    functional_form: str = "linear"  # "linear", "threshold", "interaction"
    
    def apply(self, cause_value: float, context: Dict) -> float:
        """Apply mechanism to predict effect"""
        if self.functional_form == "linear":
            effect = cause_value
            for mediator in self.mediators:
                if mediator in context:
                    effect *= context[mediator]
            return effect
        
        elif self.functional_form == "threshold":
            threshold = self.moderators.get('threshold', 0.5)
            return 1.0 if cause_value > threshold else 0.0
        
        elif self.functional_form == "interaction":
            # Multiplicative interaction
            result = cause_value
            for mod, weight in self.moderators.items():
                if mod in context:
                    result *= (1 + weight * context[mod])
            return result
        
        return cause_value


class DeepCausalGraph:
    """
    Multi-order causal graph with latent variables
    """
    
    def __init__(self):
        self.observable_graph = nx.DiGraph()
        self.latent_graph = nx.DiGraph()
        self.mechanisms: Dict[Tuple[str, str], CausalMechanism] = {}
        self.latent_variables: Dict[str, LatentVariable] = {}
        self.causal_orders: Dict[Tuple[str, str], CausalOrder] = {}
        
    def add_observed_edge(self, cause: str, effect: str, 
                         strength: float = 0.5):
        """Add first-order causal edge"""
        self.observable_graph.add_edge(cause, effect, weight=strength)
        self.causal_orders[(cause, effect)] = CausalOrder.FIRST_ORDER
        
    def infer_latent_causes(self, observations: List[Dict],
                           min_confidence: float = 0.3) -> List[LatentVariable]:
        """
        Infer latent variables from patterns in observations
        
        Example: If A→B and A→C always happen together, 
        maybe there's latent L causing both
        """
        new_latents = []
        
        # Find common causes with similar effects
        for node in self.observable_graph.nodes():
            successors = list(self.observable_graph.successors(node))
            
            if len(successors) >= 2:
                # Check if effects are correlated
                correlations = self._compute_effect_correlations(
                    observations, successors
                )
                
                if correlations > 0.7:
                    # Infer latent common cause
                    latent_name = f"latent_{node}_driver"
                    latent = LatentVariable(
                        name=latent_name,
                        inferred_from=[node] + successors,
                        distribution=self._estimate_latent_distribution(
                            observations, node, successors
                        ),
                        confidence=correlations * 0.8
                    )
                    
                    self.latent_variables[latent_name] = latent
                    self.latent_graph.add_node(latent_name, latent=latent)
                    self.latent_graph.add_edge(latent_name, node)
                    
                    for succ in successors:
                        self.latent_graph.add_edge(latent_name, succ)
                    
                    new_latents.append(latent)
        
        return new_latents
    
    def _compute_effect_correlations(self, observations: List[Dict],
                                    effects: List[str]) -> float:
        """Compute average correlation between effects"""
        if len(observations) < 3 or len(effects) < 2:
            return 0.0
        
        correlations = []
        for i, e1 in enumerate(effects):
            for e2 in effects[i+1:]:
                vals1 = [obs.get(e1, 0) for obs in observations if e1 in obs and e2 in obs]
                vals2 = [obs.get(e2, 0) for obs in observations if e1 in obs and e2 in obs]
                
                if len(vals1) >= 3:
                    try:
                        corr = np.corrcoef(vals1, vals2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except:
                        pass
        
        return np.mean(correlations) if correlations else 0.0
    
    def _estimate_latent_distribution(self, observations: List[Dict],
                                     cause: str, effects: List[str]) -> np.ndarray:
        """Estimate distribution of latent variable"""
        # Simple heuristic: latent value is weighted combination
        # of cause and average effect
        samples = []
        for obs in observations:
            if cause in obs:
                cause_val = obs[cause]
                effect_vals = [obs.get(e, 0) for e in effects if e in obs]
                avg_effect = np.mean(effect_vals) if effect_vals else 0
                
                # Latent is "residual" - unexplained variance
                latent_val = 0.6 * cause_val + 0.4 * avg_effect
                samples.append(latent_val)
        
        if not samples:
            return np.array([0.5])
        
        return np.array(samples)
    
    def discover_second_order_causality(self) -> List[Tuple[str, str, str]]:
        """
        Discover chains: A → B → C
        
        Returns:
            List of (A, B, C) chains
        """
        chains = []
        
        for node_a in self.observable_graph.nodes():
            for node_b in self.observable_graph.successors(node_a):
                for node_c in self.observable_graph.successors(node_b):
                    # Found A → B → C
                    chains.append((node_a, node_b, node_c))
                    
                    # Mark as second-order
                    self.causal_orders[(node_a, node_c)] = CausalOrder.SECOND_ORDER
                    
                    # Create mechanism
                    mechanism = CausalMechanism(
                        cause=node_a,
                        effect=node_c,
                        mechanism_type="mediated",
                        mediators=[node_b],
                        functional_form="interaction"
                    )
                    self.mechanisms[(node_a, node_c)] = mechanism
        
        return chains
    
    def discover_third_order_causality(self) -> List[Tuple[str, str, str, str]]:
        """
        Discover meta-causality: A → B affects how C → D
        
        This is the key for ARC level 8-10 puzzles!
        """
        meta_causal = []
        
        # Find pairs of causal relationships
        edges = list(self.observable_graph.edges())
        
        for (a, b), (c, d) in itertools.combinations(edges, 2):
            # Check if A→B modulates C→D
            if self._test_modulation(a, b, c, d):
                meta_causal.append((a, b, c, d))
                
                # Create third-order mechanism
                mechanism = CausalMechanism(
                    cause=f"{a}_causing_{b}",
                    effect=f"{c}_causing_{d}",
                    mechanism_type="modulating",
                    functional_form="interaction",
                    moderators={a: 0.5, b: 0.5}
                )
                self.mechanisms[(f"{a}_causing_{b}", f"{c}_causing_{d}")] = mechanism
                self.causal_orders[(a, d)] = CausalOrder.THIRD_ORDER
        
        return meta_causal
    
    def _test_modulation(self, a: str, b: str, c: str, d: str) -> bool:
        """Test if A→B modulates C→D"""
        # Heuristic: check if correlation of C→D changes when A→B is active
        # For now, use structural heuristic
        return a != c and b != d and a != d and b != c
    
    def counterfactual_inference(self, intervention: Dict[str, float],
                                order_limit: CausalOrder = CausalOrder.HIGHER_ORDER
                               ) -> Dict[str, float]:
        """
        Perform counterfactual reasoning at specified order
        
        Args:
            intervention: Variables to intervene on {var: value}
            order_limit: Maximum causal order to consider
            
        Returns:
            Predicted effects
        """
        effects = {}
        
        # First-order effects
        if order_limit.value >= 1:
            for var, val in intervention.items():
                # Skip if variable not in graph
                if var not in self.observable_graph:
                    continue
                for succ in self.observable_graph.successors(var):
                    if (var, succ) in self.mechanisms:
                        mech = self.mechanisms[(var, succ)]
                        effects[succ] = mech.apply(val, {})
                    else:
                        effects[succ] = val * 0.8  # Default attenuation
        
        # Second-order effects
        if order_limit.value >= 2:
            for var, val in intervention.items():
                if var not in self.observable_graph:
                    continue
                chains = self._find_chains_from(var)
                for (a, b, c) in chains:
                    if b in effects:
                        # Mediated effect
                        mediated_effect = effects[b] * 0.7
                        effects[c] = effects.get(c, 0) + mediated_effect
        
        # Third-order (modulation) effects
        if order_limit.value >= 3:
            # Check if any causal relationships are modulated
            for var in intervention:
                for (cause, mech) in self.mechanisms.items():
                    if mech.mechanism_type == "modulating":
                        if var in mech.moderators:
                            # This relationship strength is modulated
                            modulation = intervention[var] * mech.moderators[var]
                            # Would need actual implementation
                            pass
        
        return effects
    
    def _find_chains_from(self, start: str) -> List[Tuple[str, str, str]]:
        """Find all A→B→C chains starting from A"""
        chains = []
        for b in self.observable_graph.successors(start):
            for c in self.observable_graph.successors(b):
                chains.append((start, b, c))
        return chains
    
    def explain_causal_path(self, cause: str, effect: str) -> str:
        """
        Generate human-readable explanation of causal path
        """
        try:
            # Check if nodes exist first
            if cause not in self.observable_graph or effect not in self.observable_graph:
                return f"Nodes not in causal graph: {cause} or {effect}"
            
            # Find path in graph
            path = nx.shortest_path(self.observable_graph, cause, effect)
            
            if len(path) == 2:
                return f"{cause} directly causes {effect}"
            
            explanation = f"{cause} causes {effect} through:"
            for i in range(len(path) - 1):
                edge_order = self.causal_orders.get(
                    (path[i], path[i+1]), CausalOrder.FIRST_ORDER
                )
                explanation += f"\n  → {path[i]} ({edge_order.name}) → {path[i+1]}"
            
            return explanation
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return f"No causal path found from {cause} to {effect}"


class DeepCausalEngine:
    """
    Main engine for deep causal reasoning in ARC-AGI-3
    """
    
    def __init__(self):
        self.graph = DeepCausalGraph()
        self.observation_buffer: List[Dict] = []
        self.max_buffer_size = 100
        
    def observe(self, state: Dict, context: Dict = None):
        """Record observation for causal learning"""
        obs = state.copy()
        if context:
            obs['_context'] = context
        
        self.observation_buffer.append(obs)
        
        if len(self.observation_buffer) > self.max_buffer_size:
            self.observation_buffer = self.observation_buffer[-self.max_buffer_size:]
    
    def learn_structure(self, min_observations: int = 10):
        """Learn deep causal structure from observations"""
        if len(self.observation_buffer) < min_observations:
            return {}
        
        results = {
            'latents_inferred': 0,
            'second_order_found': 0,
            'third_order_found': 0
        }
        
        # 1. Infer latent variables
        latents = self.graph.infer_latent_causes(self.observation_buffer)
        results['latents_inferred'] = len(latents)
        
        # 2. Discover second-order causality
        chains = self.graph.discover_second_order_causality()
        results['second_order_found'] = len(chains)
        
        # 3. Discover third-order causality (meta-causality)
        meta = self.graph.discover_third_order_causality()
        results['third_order_found'] = len(meta)
        
        return results
    
    def deep_intervention(self, intervention: Dict[str, float],
                         max_order: int = 3) -> Dict[str, Any]:
        """
        Perform deep intervention considering all causal orders
        
        This is the key for ARC puzzles requiring hidden reasoning!
        """
        order_enum = {
            1: CausalOrder.FIRST_ORDER,
            2: CausalOrder.SECOND_ORDER,
            3: CausalOrder.THIRD_ORDER
        }.get(max_order, CausalOrder.HIGHER_ORDER)
        
        effects = self.graph.counterfactual_inference(intervention, order_enum)
        
        # Also check latent effects
        latent_effects = {}
        for latent_name, latent in self.graph.latent_variables.items():
            # Check if intervention affects this latent's children
            for parent in latent.inferred_from:
                if parent in intervention:
                    # Update latent
                    latent_effects[latent_name] = intervention[parent] * latent.confidence
        
        return {
            'observable_effects': effects,
            'latent_effects': latent_effects,
            'explanation': self._generate_explanation(intervention, effects)
        }
    
    def _generate_explanation(self, intervention: Dict, effects: Dict) -> str:
        """Generate explanation of intervention effects"""
        explanations = []
        
        for var, val in intervention.items():
            for effect, eff_val in effects.items():
                path_explanation = self.graph.explain_causal_path(var, effect)
                explanations.append(f"{path_explanation}: {eff_val:.2f}")
        
        return "; ".join(explanations[:3])  # Top 3
    
    def query_causal_structure(self) -> Dict:
        """Get summary of learned causal structure"""
        return {
            'observable_nodes': len(self.graph.observable_graph.nodes()),
            'observable_edges': len(self.graph.observable_graph.edges()),
            'latent_variables': len(self.graph.latent_variables),
            'mechanisms_learned': len(self.graph.mechanisms),
            'max_causal_order': max(
                (o.value for o in self.graph.causal_orders.values()),
                default=1
            ),
            'latent_variables_detail': [
                {'name': lv.name, 'confidence': lv.confidence, 
                 'inferred_from': lv.inferred_from}
                for lv in self.graph.latent_variables.values()
            ]
        }
