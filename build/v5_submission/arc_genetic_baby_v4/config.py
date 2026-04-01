"""Configuration management for ARC Genetic Baby V4."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

import yaml


@dataclass
class PerceptionConfig:
    """Configuration for Predictive Perception layer."""
    num_levels: int = 3
    level1_hidden_dim: int = 128
    level2_hidden_dim: int = 256
    level3_hidden_dim: int = 512
    prediction_threshold: float = 0.1
    learning_rate: float = 0.001


@dataclass
class ActiveInferenceConfig:
    """Configuration for Active Inference engine."""
    horizon: int = 3  # Planning horizon
    num_samples: int = 100  # Samples for action evaluation
    temperature: float = 1.0  # Softmax temperature
    gamma: float = 0.99  # Discount factor
    complexity_weight: float = 0.1  # Weight for complexity term in Free Energy
    free_energy_threshold: float = -0.5  # Threshold for triggering program synthesis


@dataclass
class ProgramSynthesisConfig:
    """Configuration for Evolutionary Program Synthesis."""
    population_size: int = 100
    generations: int = 50
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 7
    hall_of_fame_size: int = 5
    parsimony_coefficient: float = 0.01  # Occam's razor weight
    max_tree_depth: int = 10


@dataclass
class AnalogyConfig:
    """Configuration for Structural Analogy Engine."""
    max_graph_nodes: int = 1000
    structural_match_threshold: float = 0.7
    min_common_relations: int = 3
    analogy_cache_size: int = 100


@dataclass
class SleepConfig:
    """Configuration for Sleep Consolidation."""
    consolidation_interval: int = 100  # Experiences before consolidation
    compression_ratio: float = 0.1
    similarity_threshold: float = 0.8
    max_schemas: int = 1000


@dataclass
class EnsembleConfig:
    """Configuration for Genetic Ensemble."""
    population_size: int = 10
    diversity_threshold: float = 0.3
    exploration_rate: float = 0.1
    voting_method: str = "weighted"  # weighted, majority, borda


@dataclass
class MemoryConfig:
    """Configuration for Relational Memory."""
    max_episodic_memories: int = 10000
    max_semantic_schemas: int = 1000
    forgetting_rate: float = 0.01
    retrieval_k: int = 5


@dataclass
class AgentConfig:
    """Complete configuration for ARC Genetic Baby V4."""
    
    # ARC-AGI-3 specifications
    grid_size: int = 64
    num_colors: int = 16
    max_actions: int = 1000
    
    # Cognitive layer configs
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    active_inference: ActiveInferenceConfig = field(default_factory=ActiveInferenceConfig)
    program_synthesis: ProgramSynthesisConfig = field(default_factory=ProgramSynthesisConfig)
    analogy: AnalogyConfig = field(default_factory=AnalogyConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Development
    developmental_stages: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    checkpoint_interval: int = 100
    
    # Paths
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
    
    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        data = dataclasses.asdict(self)
        
        # Convert Path objects to strings
        if 'checkpoint_dir' in data:
            data['checkpoint_dir'] = str(data['checkpoint_dir'])
        if 'log_dir' in data:
            data['log_dir'] = str(data['log_dir'])
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        assert self.grid_size > 0, "Grid size must be positive"
        assert self.num_colors > 0, "Number of colors must be positive"
        assert self.perception.num_levels >= 1, "Must have at least 1 perception level"
        assert self.ensemble.population_size >= 2, "Ensemble needs at least 2 agents"
        assert self.program_synthesis.population_size >= 10, "Population too small"
        return True
