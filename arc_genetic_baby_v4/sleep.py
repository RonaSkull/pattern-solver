"""Layer 5: Sleep Consolidation + Genetic Ensemble.

Implements:
    1. Sleep Consolidation: Memory replay and schema abstraction inspired by
       human memory consolidation (hippocampus -> cortex)
    2. Genetic Ensemble: Multiple agents with different "DNA" strategies that
       vote on actions, ensuring robustness through diversity

References:
    - Rasch, B., & Born, J. (2013). About sleep's role in memory.
    - Breiman, L. (2001). Random forests. Machine learning.
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import random

from .config import SleepConfig, EnsembleConfig


# ==================== SLEEP CONSOLIDATION ====================

@dataclass
class Experience:
    """A single experience episode."""
    state: np.ndarray
    action: str
    next_state: np.ndarray
    success: bool
    reward: float = 0.0
    timestamp: int = 0
    problem_id: str = ""
    
    def to_vector(self) -> np.ndarray:
        """Convert experience to feature vector for clustering."""
        # Combine state features
        state_flat = self.state.flatten()[:100]  # Limit size
        next_flat = self.next_state.flatten()[:100]
        
        # Pad to consistent size
        state_padded = np.pad(state_flat, (0, 100 - len(state_flat)))
        next_padded = np.pad(next_flat, (0, 100 - len(next_flat)))
        
        # Add metadata
        meta = np.array([
            1.0 if self.success else 0.0,
            self.reward,
            float(ord(self.action[0])) if self.action else 0.0
        ])
        
        return np.concatenate([state_padded, next_padded, meta])


@dataclass
class CognitiveSchema:
    """
    An abstract schema extracted from multiple experiences.
    
    Represents compressed, generalizable knowledge.
    """
    schema_id: str
    pattern_signature: Tuple  # Identifying pattern
    action_template: str  # Generalized action
    conditions: List[Callable[[np.ndarray], bool]]  # When to apply
    success_rate: float = 0.0
    usage_count: int = 0
    parent_schemas: List[str] = field(default_factory=list)
    
    def matches(self, state: np.ndarray) -> bool:
        """Check if state matches this schema's conditions."""
        for condition in self.conditions:
            try:
                if not condition(state):
                    return False
            except Exception:
                return False
        return True
    
    def apply(self, state: np.ndarray) -> Optional[str]:
        """Apply schema to generate action."""
        if self.matches(state):
            return self.action_template
        return None


class SleepConsolidation:
    """
    Implements memory consolidation inspired by sleep processes.
    
    Key processes:
        1. Replay: Reactivate recent experiences
        2. Pattern extraction: Find common structures
        3. Abstraction: Create schemas from patterns
        4. Pruning: Remove redundant details
    
    Based on complementary learning systems theory:
        - Hippocampus: Fast learning, episodic memories
        - Cortex: Slow learning, semantic schemas
        - Sleep transfers from hippocampus to cortex
    """
    
    def __init__(self, config: SleepConfig = None):
        self.config = config or SleepConfig()
        self.experience_buffer: List[Experience] = []
        self.unconsolidated: List[Experience] = []
        self.consolidated_schemas: List[CognitiveSchema] = []
        self.schema_index = 0
        
    def add_experience(self, exp: Experience):
        """Add experience to buffer for consolidation."""
        self.experience_buffer.append(exp)
        self.unconsolidated.append(exp)
        
        # Check if consolidation needed
        if len(self.unconsolidated) >= self.config.consolidation_interval:
            self.consolidate()
    
    def consolidate(self, experiences: List[Experience] = None) -> List[CognitiveSchema]:
        """
        Consolidate experiences into abstract schemas.
        
        Steps:
            1. Cluster experiences by structural similarity
            2. Extract common patterns from each cluster
            3. Create schemas from patterns
            4. Prune redundant schemas
        """
        if experiences is None:
            experiences = self.experience_buffer
            self.experience_buffer = []  # Clear buffer
            self.unconsolidated = []  # Clear unconsolidated too
        
        if len(experiences) < 10:
            return []
        
        new_schemas = []
        
        # Step 1: Cluster experiences
        clusters = self._cluster_experiences(experiences)
        
        # Step 2 & 3: Extract patterns and create schemas
        for cluster_id, cluster_exps in clusters.items():
            if len(cluster_exps) < 3:
                continue
                
            schema = self._extract_schema(cluster_exps)
            if schema:
                new_schemas.append(schema)
        
        # Step 4: Prune and merge
        new_schemas = self._prune_schemas(new_schemas)
        new_schemas = self._merge_with_existing(new_schemas)
        
        # Update schema library
        self.consolidated_schemas.extend(new_schemas)
        
        # Keep only top schemas
        self.consolidated_schemas = sorted(
            self.consolidated_schemas,
            key=lambda s: s.success_rate * s.usage_count,
            reverse=True
        )[:self.config.max_schemas]
        
        return new_schemas
    
    def _cluster_experiences(self, experiences: List[Experience]) -> Dict[int, List[Experience]]:
        """Cluster experiences by structural similarity."""
        if len(experiences) < 10:
            return {0: experiences}
        
        # Convert to feature vectors
        vectors = np.array([exp.to_vector() for exp in experiences])
        
        # Normalize
        vectors = (vectors - vectors.mean(axis=0)) / (vectors.std(axis=0) + 1e-8)
        
        # Hierarchical clustering
        distance_matrix = pdist(vectors, metric='euclidean')
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Determine number of clusters
        n_clusters = max(2, int(len(experiences) * self.config.compression_ratio))
        
        # Assign clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Group by cluster
        clusters = defaultdict(list)
        for exp, label in zip(experiences, cluster_labels):
            clusters[label].append(exp)
        
        return dict(clusters)
    
    def _extract_schema(self, experiences: List[Experience]) -> Optional[CognitiveSchema]:
        """Extract common pattern from cluster of experiences."""
        if not experiences:
            return None
        
        # Find common action
        actions = [exp.action for exp in experiences]
        action_counts = defaultdict(int)
        for a in actions:
            action_counts[a] += 1
        
        most_common_action = max(action_counts.keys(), key=lambda k: action_counts[k])
        action_frequency = action_counts[most_common_action] / len(experiences)
        
        # Only create schema if action is consistent
        if action_frequency < 0.6:
            return None
        
        # Calculate success rate
        successes = sum(1 for exp in experiences if exp.success)
        success_rate = successes / len(experiences)
        
        # Extract common state patterns
        states = [exp.state for exp in experiences]
        common_pattern = self._find_common_pattern(states)
        
        # Create conditions based on pattern
        conditions = self._create_conditions(common_pattern)
        
        # Create schema
        self.schema_index += 1
        schema = CognitiveSchema(
            schema_id=f"schema_{self.schema_index}",
            pattern_signature=common_pattern,
            action_template=most_common_action,
            conditions=conditions,
            success_rate=success_rate,
            usage_count=0
        )
        
        return schema
    
    def _find_common_pattern(self, states: List[np.ndarray]) -> Tuple:
        """Find common structural pattern across states."""
        # Simplified: compute statistics
        if not states:
            return ()
        
        states_array = np.array([s.flatten()[:100] for s in states])
        
        # Pattern: mean and std of state features
        mean_pattern = tuple(np.round(states_array.mean(axis=0)[:10], 2))
        
        return mean_pattern
    
    def _create_conditions(self, pattern: Tuple) -> List[Callable]:
        """Create matching conditions from pattern."""
        conditions = []
        
        # Example condition: check if state has similar feature distribution
        def feature_condition(state):
            state_features = state.flatten()[:len(pattern)]
            similarity = np.corrcoef(state_features, pattern)[0, 1]
            return similarity > self.config.similarity_threshold if not np.isnan(similarity) else False
        
        conditions.append(feature_condition)
        
        return conditions
    
    def _prune_schemas(self, schemas: List[CognitiveSchema]) -> List[CognitiveSchema]:
        """Remove redundant schemas."""
        if not schemas:
            return schemas
        
        # Sort by quality
        sorted_schemas = sorted(
            schemas,
            key=lambda s: (s.success_rate, s.usage_count),
            reverse=True
        )
        
        # Keep non-redundant schemas
        pruned = []
        for schema in sorted_schemas:
            # Check if too similar to existing
            is_redundant = False
            for kept in pruned:
                if self._schema_similarity(schema, kept) > 0.9:
                    is_redundant = True
                    break
            
            if not is_redundant:
                pruned.append(schema)
        
        return pruned
    
    def _merge_with_existing(self, new_schemas: List[CognitiveSchema]) -> List[CognitiveSchema]:
        """Merge new schemas with existing ones."""
        merged = []
        
        for new_schema in new_schemas:
            # Find similar existing schema
            similar = None
            for existing in self.consolidated_schemas:
                if self._schema_similarity(new_schema, existing) > 0.8:
                    similar = existing
                    break
            
            if similar:
                # Merge: update success rate and usage
                similar.success_rate = (
                    similar.success_rate * similar.usage_count +
                    new_schema.success_rate * 10
                ) / (similar.usage_count + 10)
                similar.usage_count += 1
            else:
                merged.append(new_schema)
        
        return merged
    
    def _schema_similarity(self, s1: CognitiveSchema, s2: CognitiveSchema) -> float:
        """Calculate similarity between two schemas."""
        # Compare action templates
        if s1.action_template != s2.action_template:
            return 0.0
        
        # Compare pattern signatures
        if s1.pattern_signature == s2.pattern_signature:
            return 1.0
        
        # Approximate similarity
        return 0.5
    
    def get_applicable_schemas(self, state: np.ndarray) -> List[CognitiveSchema]:
        """Find schemas that match current state."""
        applicable = []
        for schema in self.consolidated_schemas:
            if schema.matches(state):
                applicable.append(schema)
                schema.usage_count += 1
        
        # Sort by expected success
        applicable.sort(key=lambda s: s.success_rate, reverse=True)
        return applicable
    
    def save_checkpoint(self) -> Dict:
        """Save schemas for later use."""
        return {
            'schemas': self.consolidated_schemas,
            'schema_count': len(self.consolidated_schemas)
        }
    
    def load_checkpoint(self, data: Dict):
        """Load schemas from checkpoint."""
        self.consolidated_schemas = data.get('schemas', [])


# ==================== GENETIC ENSEMBLE ====================

@dataclass
class AgentDNA:
    """
    DNA that determines an agent's strategy and biases.
    
    Encodes:
        - Perception biases
        - Action selection preferences
        - Exploration vs exploitation tendency
    """
    exploration_rate: float = 0.1
    perception_level_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    analogy_bias: float = 0.5
    program_synthesis_bias: float = 0.5
    free_energy_threshold: float = -0.5
    voting_confidence_weight: float = 1.0
    
    def mutate(self, mutation_rate: float = 0.1):
        """Create mutated copy of DNA."""
        new_dna = AgentDNA(
            exploration_rate=np.clip(
                self.exploration_rate + np.random.normal(0, mutation_rate), 0, 1
            ),
            perception_level_weights=[
                np.clip(w + np.random.normal(0, mutation_rate), 0.1, 2.0)
                for w in self.perception_level_weights
            ],
            analogy_bias=np.clip(
                self.analogy_bias + np.random.normal(0, mutation_rate), 0, 1
            ),
            program_synthesis_bias=np.clip(
                self.program_synthesis_bias + np.random.normal(0, mutation_rate), 0, 1
            ),
            free_energy_threshold=self.free_energy_threshold + np.random.normal(0, mutation_rate),
            voting_confidence_weight=np.clip(
                self.voting_confidence_weight + np.random.normal(0, mutation_rate), 0.1, 2.0
            )
        )
        return new_dna
    
    @classmethod
    def random_dna(cls) -> 'AgentDNA':
        """Generate random DNA."""
        return cls(
            exploration_rate=np.random.uniform(0.05, 0.3),
            perception_level_weights=[np.random.uniform(0.5, 1.5) for _ in range(3)],
            analogy_bias=np.random.uniform(0.3, 0.7),
            program_synthesis_bias=np.random.uniform(0.3, 0.7),
            free_energy_threshold=np.random.uniform(-1.0, 0.0),
            voting_confidence_weight=np.random.uniform(0.5, 1.5)
        )


class GeneticBabyAgent:
    """
    A single "baby" agent with its own DNA/strategy.
    
    Part of the genetic ensemble - multiple agents with different strategies
    vote on the final action.
    """
    
    def __init__(self, dna: AgentDNA, agent_id: int):
        self.dna = dna
        self.id = agent_id
        self.history: List[Dict] = []
        self.success_rate = 0.5
        
    def propose_action(self, state: np.ndarray, available_actions: List[str],
                      beliefs: Any) -> Tuple[str, float]:
        """
        Propose an action based on this agent's DNA/strategy.
        
        Returns:
            (action, confidence)
        """
        # Simple strategy: weighted random with DNA biases
        
        # Apply exploration
        if np.random.random() < self.dna.exploration_rate:
            action = np.random.choice(available_actions)
            return action, 0.5
        
        # Use DNA to weight action preferences
        # Simplified: just pick based on agent ID for diversity
        weights = np.ones(len(available_actions))
        
        # Bias toward certain actions based on DNA
        if self.dna.analogy_bias > 0.6:
            # Prefer pattern-matching actions
            weights = np.array([
                1.5 if 'rotate' in a or 'flip' in a else 1.0
                for a in available_actions
            ])
        
        # Normalize and sample
        weights = weights / weights.sum()
        action_idx = np.random.choice(len(available_actions), p=weights)
        action = available_actions[action_idx]
        
        # Confidence based on success rate and DNA
        confidence = self.success_rate * self.dna.voting_confidence_weight
        
        return action, confidence
    
    def update_success(self, success: bool):
        """Update agent's success rate."""
        self.success_rate = 0.9 * self.success_rate + 0.1 * (1.0 if success else 0.0)


class GeneticEnsemble:
    """
    Ensemble of agents with different DNA that vote on actions.
    
    Genetic diversity ensures robustness against overfitting and
    provides multiple problem-solving strategies.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.agents: List[GeneticBabyAgent] = []
        self._init_population()
        
        # Diversity tracking
        self.diversity_history: List[float] = []
        
    def _init_population(self):
        """Initialize diverse population of agents."""
        for i in range(self.config.population_size):
            dna = AgentDNA.random_dna()
            agent = GeneticBabyAgent(dna, agent_id=i)
            self.agents.append(agent)
    
    def vote_action(self, state: np.ndarray, available_actions: List[str],
                   beliefs: Any = None, parallel: bool = True) -> Tuple[str, Dict]:
        """
        Have all agents vote and return winning action.
        
        Voting methods:
            - weighted: Weight by agent confidence and success rate
            - majority: Simple majority vote
            - borda: Borda count ranking
            
        Args:
            parallel: If True, use joblib for parallel voting (1000 FPS optimization)
        """
        # Collect votes
        votes: Dict[str, float] = defaultdict(float)
        agent_votes: Dict[int, Tuple[str, float]] = {}
        
        if parallel and len(self.agents) > 4:
            # Parallel voting with joblib for performance
            try:
                from joblib import Parallel, delayed
                
                def get_vote(agent, state, actions, beliefs):
                    action, confidence = agent.propose_action(state, actions, beliefs)
                    return agent.id, action, confidence, agent.success_rate * confidence * agent.dna.voting_confidence_weight
                
                results = Parallel(n_jobs=4)(
                    delayed(get_vote)(agent, state, available_actions, beliefs) 
                    for agent in self.agents
                )
                
                for agent_id, action, confidence, weight in results:
                    agent_votes[agent_id] = (action, confidence)
                    votes[action] += weight
                    
            except ImportError:
                # Fallback to sequential if joblib not available
                parallel = False
        
        if not parallel:
            # Sequential voting (fallback)
            for agent in self.agents:
                action, confidence = agent.propose_action(state, available_actions, beliefs)
                agent_votes[agent.id] = (action, confidence)
                
                # Weight by agent's success rate and confidence
                weight = agent.success_rate * confidence * agent.dna.voting_confidence_weight
                votes[action] += weight
        
        # Calculate diversity
        diversity = self._calculate_diversity(agent_votes)
        diversity = max(0.0, min(1.0, diversity))  # Clamp to [0, 1]
        self.diversity_history.append(diversity)
        
        # Check diversity threshold
        if diversity < self.config.diversity_threshold:
            # Force exploration: add random noise to votes
            for action in available_actions:
                votes[action] += np.random.uniform(0, self.config.exploration_rate)
        
        # Select winner
        if self.config.voting_method == "weighted":
            winning_action = max(votes.keys(), key=lambda a: votes[a])
        elif self.config.voting_method == "majority":
            # Count occurrences
            action_counts = defaultdict(int)
            for agent in self.agents:
                action, _ = agent.propose_action(state, available_actions, beliefs)
                action_counts[action] += 1
            winning_action = max(action_counts.keys(), key=lambda a: action_counts[a])
        else:  # borda
            winning_action = max(votes.keys(), key=lambda a: votes[a])
        
        # Calculate vote statistics
        total_votes = sum(votes.values())
        vote_shares = {a: v / total_votes for a, v in votes.items()}
        
        info = {
            'winning_action': winning_action,
            'vote_shares': vote_shares,
            'diversity': diversity,
            'num_agents': len(self.agents),
            'forced_exploration': diversity < self.config.diversity_threshold
        }
        
        return winning_action, info
    
    def weighted_vote(self, proposals: List[Tuple[str, float, str]]) -> str:
        """
        proposals: list of (action, confidence, agent_dna_hash)
        Voto ponderado por confiança + bonus de diversidade
        """
        from collections import defaultdict
        votes = defaultdict(float)
        
        # Agrupar por DNA para medir diversidade
        dna_groups = defaultdict(list)
        for action, conf, dna_hash in proposals:
            dna_groups[dna_hash].append((action, conf))
        
        # Calcular peso por diversidade do grupo
        total_proposals = len(proposals)
        for dna_hash, group in dna_groups.items():
            group_size = len(group)
            diversity_bonus = 1.0 + 0.3 * (1 - group_size / total_proposals)
            
            for action, conf in group:
                votes[action] += conf * diversity_bonus
        
        return max(votes.keys(), key=lambda a: votes[a])
    
    def _calculate_diversity(self, agent_votes: Dict[int, Tuple[str, float]]) -> float:
        """Calculate diversity of agent proposals."""
        actions = [vote[0] for vote in agent_votes.values()]
        
        # Entropy of action distribution
        unique, counts = np.unique(actions, return_counts=True)
        probs = counts / len(actions)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Normalize by max entropy
        max_entropy = np.log(len(self.agents))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def evolve_population(self, fitness_scores: List[float]):
        """
        Evolve population based on fitness scores.
        
        Replace low-performing agents with mutated versions of high-performers.
        """
        # Pair agents with fitness
        agent_fitness = list(zip(self.agents, fitness_scores))
        
        # Sort by fitness
        agent_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top half, replace bottom half with mutations
        n_keep = len(self.agents) // 2
        
        top_agents = [af[0] for af in agent_fitness[:n_keep]]
        
        # Create new agents from top performers
        new_agents = top_agents.copy()
        for i in range(len(self.agents) - n_keep):
            # Select parent from top half
            parent = np.random.choice(top_agents)
            # Mutate
            new_dna = parent.dna.mutate(mutation_rate=0.2)
            new_agent = GeneticBabyAgent(new_dna, agent_id=len(new_agents) + i)
            new_agents.append(new_agent)
        
        self.agents = new_agents
    
    def update_agent_success(self, agent_id: int, success: bool):
        """Update success rate for specific agent."""
        for agent in self.agents:
            if agent.id == agent_id:
                agent.update_success(success)
                break
    
    def get_agent_stats(self) -> Dict:
        """Get statistics about the ensemble."""
        return {
            'population_size': len(self.agents),
            'mean_success_rate': np.mean([a.success_rate for a in self.agents]),
            'mean_diversity': np.mean(self.diversity_history) if self.diversity_history else 0,
            'exploration_rates': [a.dna.exploration_rate for a in self.agents]
        }
