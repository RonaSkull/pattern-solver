"""ARC-AGI-3 Genetic Baby V5 - 6 Critical Gaps Implementation."""

__version__ = "5.0.0"
__all__ = [
    # V4 Base
    "ARCGeneticBabyV4",
    "ARCGeneticBabyV5",
    "PredictivePerception",
    "ActiveInferenceAgent", 
    "EvolutionaryProgramSynthesizer",
    "StructuralAnalogyEngine",
    "SleepConsolidation",
    "GeneticEnsemble",
    "RelationalMemory",
    "AgentConfig",
    # V5 Critical Gaps
    "CausalDiscoveryEngine",
    "SymbolicAbstractionModule",
    "CounterfactualEngine",
    "HierarchicalPlanner",
    "LearnedAttentionMechanism",
    "MetaLearner",
]

from .agent import ARCGeneticBabyV4
from .agent_v5 import ARCGeneticBabyV5
from .config import AgentConfig
from .perception import PredictivePerception
from .active_inference import ActiveInferenceAgent
from .program_synthesis import EvolutionaryProgramSynthesizer
from .analogy import StructuralAnalogyEngine
from .sleep import SleepConsolidation, GeneticEnsemble
from .memory import RelationalMemory
from .causal_discovery import CausalDiscoveryEngine
from .symbolic_abstraction import SymbolicAbstractionModule
from .counterfactual import CounterfactualEngine
from .planner import HierarchicalPlanner
from .attention import LearnedAttentionMechanism
from .meta_learning import MetaLearner
