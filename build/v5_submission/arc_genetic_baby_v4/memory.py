"""Relational Memory system for episodic and semantic storage."""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import MemoryConfig
from .sleep import Experience, CognitiveSchema


@dataclass
class EpisodicMemory:
    """Memory of specific experiences."""
    experience: Experience
    embedding: np.ndarray
    timestamp: int
    importance: float = 1.0


@dataclass
class Problem:
    """Representation of a problem/task."""
    id: str
    grid: np.ndarray
    available_actions: List[str]
    solution: Optional[str] = None
    difficulty: float = 0.5
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"problem_{hash(self.grid.tobytes()) % 10000}"


class RelationalMemory:
    """
    Memory system with episodic and semantic components.
    
    Implements complementary learning systems:
        - Episodic: Fast storage of specific experiences
        - Semantic: Slow storage of abstract schemas
        - Retrieval: Similarity-based search
    """
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        
        # Episodic memory (hippocampus-like)
        self.episodic: deque = deque(maxlen=self.config.max_episodic_memories)
        
        # Semantic memory (cortex-like)
        self.semantic_schemas: List[CognitiveSchema] = []
        self.solved_problems: Dict[str, Problem] = {}
        
        # Embedding index for fast retrieval
        self.embeddings: List[np.ndarray] = []
        self.nn_index: Optional[NearestNeighbors] = None
        
        # Unconsolidated experiences (waiting for sleep)
        self.unconsolidated: List[Experience] = []
        
        self.timestamp = 0
        
    def store(self, experience: Experience):
        """Store experience in episodic memory."""
        self.timestamp += 1
        
        # Create embedding
        embedding = experience.to_vector()
        
        # Store
        episodic = EpisodicMemory(
            experience=experience,
            embedding=embedding,
            timestamp=self.timestamp
        )
        
        self.episodic.append(episodic)
        self.unconsolidated.append(experience)
        
        # Update index
        self._update_index()
        
    def store_problem(self, problem: Problem):
        """Store a solved problem."""
        self.solved_problems[problem.id] = problem
        
    def retrieve_similar(self, state: np.ndarray, k: int = None) -> List[Problem]:
        """Retrieve similar problems from memory."""
        if k is None:
            k = self.config.retrieval_k
            
        if len(self.episodic) == 0 and len(self.solved_problems) == 0:
            return []
        
        # Create query embedding
        query = np.zeros(203)  # Match experience vector size
        state_flat = state.flatten()[:100]
        query[:len(state_flat)] = state_flat
        
        # Find similar episodes
        similar_experiences = []
        
        if len(self.episodic) > 0:
            embeddings = np.array([e.embedding for e in self.episodic])
            
            # Calculate similarities
            similarities = np.dot(embeddings, query) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query) + 1e-10
            )
            
            # Get top k
            top_k = np.argsort(similarities)[-k:][::-1]
            
            for idx in top_k:
                exp = list(self.episodic)[idx].experience
                # Convert to problem format
                problem = Problem(
                    id=f"ep_{idx}",
                    grid=exp.state,
                    available_actions=[exp.action],
                    solution=exp.action
                )
                similar_experiences.append(problem)
        
        # Add solved problems
        for prob in list(self.solved_problems.values())[:k]:
            similar_experiences.append(prob)
        
        return similar_experiences[:k]
    
    def get_recent_examples(self, n: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get recent input-output examples for program synthesis."""
        examples = []
        
        for episodic in list(self.episodic)[-n:]:
            exp = episodic.experience
            examples.append((exp.state, exp.next_state))
        
        return examples
    
    def add_schemas(self, schemas: List[CognitiveSchema]):
        """Add consolidated schemas to semantic memory."""
        self.semantic_schemas.extend(schemas)
        
        # Keep only best schemas
        self.semantic_schemas = sorted(
            self.semantic_schemas,
            key=lambda s: s.success_rate * s.usage_count,
            reverse=True
        )[:self.config.max_semantic_schemas]
    
    def get_relevant_schemas(self, state: np.ndarray) -> List[CognitiveSchema]:
        """Get schemas applicable to current state."""
        relevant = []
        for schema in self.semantic_schemas:
            if schema.matches(state):
                relevant.append(schema)
        return relevant
    
    def _update_index(self):
        """Update nearest neighbors index."""
        if len(self.episodic) < 10:
            return
            
        embeddings = np.array([e.embedding for e in self.episodic])
        
        self.nn_index = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
        self.nn_index.fit(embeddings)
    
    def forget(self):
        """Apply forgetting to old memories."""
        # Episodic memories are automatically forgotten via deque maxlen
        
        # Decay importance of schemas
        for schema in self.semantic_schemas:
            if schema.usage_count > 0:
                schema.usage_count = max(0, schema.usage_count - 1)
    
    def save_checkpoint(self) -> Dict:
        """Save memory state."""
        return {
            'episodic_count': len(self.episodic),
            'unconsolidated_count': len(self.unconsolidated),
            'schemas': self.semantic_schemas,
            'solved_problems': list(self.solved_problems.keys())
        }
    
    def load_checkpoint(self, data: Dict):
        """Load memory state."""
        self.semantic_schemas = data.get('schemas', [])
        self.unconsolidated = []
