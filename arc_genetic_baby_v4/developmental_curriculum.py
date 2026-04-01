"""
developmental_curriculum.py

Developmental Curriculum Module para ARC-AGI-3 V6

Implementa estagios de desenvolvimento cognitivo com phase transitions.
Inspiracao: Bebes nao aprendem tudo de uma vez — passam por estagios previsiveis
onde novas capacidades sao desbloqueadas progressivamente.

Version: 6.2.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class DevelopmentalStage(Enum):
    """Estagios de desenvolvimento cognitivo inspirados em Piaget."""
    PRENATAL = auto()
    SENSORIMOTOR = auto()
    PRE_OPERATIONAL = auto()
    CONCRETE_OPERATIONAL = auto()
    FORMAL_OPERATIONAL = auto()
    META_COGNITIVE = auto()


@dataclass
class StageConfig:
    """Configuracao de um estagio de desenvolvimento"""
    stage: DevelopmentalStage
    name: str
    description: str
    enabled_modules: List[str]
    mastery_threshold: float
    min_experiences: int
    disabled_modules: List[str] = field(default_factory=list)
    mastery_weights: Dict[str, float] = field(default_factory=lambda: {
        'confidence': 0.4,
        'success_rate': 0.3,
        'learning_progress': 0.2,
        'generalization': 0.1
    })
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DevelopmentalMetrics:
    """Metricas de progresso dentro de um estagio"""
    stage: DevelopmentalStage
    total_experiences: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    module_confidence: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    learning_progress: float = 0.0
    generalization_score: float = 0.0
    recent_successes: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=50))
    
    @property
    def success_rate(self) -> float:
        total = self.successful_actions + self.failed_actions
        return self.successful_actions / max(1, total)
    
    @property
    def avg_confidence(self) -> float:
        if not self.module_confidence:
            return 0.0
        return np.mean(list(self.module_confidence.values()))
    
    def compute_mastery(self, weights: Dict[str, float] = None) -> float:
        weights = weights or {
            'confidence': 0.4,
            'success_rate': 0.3,
            'learning_progress': 0.2,
            'generalization': 0.1
        }
        metrics = {
            'confidence': self.avg_confidence,
            'success_rate': self.success_rate,
            'learning_progress': np.clip(self.learning_progress, 0, 1),
            'generalization': self.generalization_score
        }
        mastery = sum(weights.get(k, 0) * v for k, v in metrics.items())
        return np.clip(mastery, 0, 1)
    
    def add_experience(self, success: bool, error: float = None):
        self.total_experiences += 1
        if success:
            self.successful_actions += 1
        else:
            self.failed_actions += 1
        self.recent_successes.append(1 if success else 0)
        if error is not None:
            self.recent_errors.append(error)
            if len(self.recent_errors) >= 10:
                recent = list(self.recent_errors)[-5:]
                older = list(self.recent_errors)[-10:-5]
                if np.mean(older) > 0:
                    self.learning_progress = 1.0 - (np.mean(recent) / np.mean(older))


class DevelopmentalCurriculum:
    """Gerenciador principal do curriculo de desenvolvimento."""
    
    DEFAULT_STAGES = {
        DevelopmentalStage.PRENATAL: StageConfig(
            stage=DevelopmentalStage.PRENATAL,
            name="Pre-Natal",
            description="Configuracao inicial, modulos basicos",
            enabled_modules=['attention', 'perception'],
            disabled_modules=['causal_discovery', 'symbolic_abstraction', 'counterfactual',
                            'planner', 'meta_learning', 'deep_causal', 'high_order_symbolic',
                            'metacognition', 'productive_composition', 'natural_instruction'],
            mastery_threshold=0.8,
            min_experiences=10,
            hyperparameters={'curiosity_weight': 0.1, 'exploration_epsilon': 0.3}
        ),
        DevelopmentalStage.SENSORIMOTOR: StageConfig(
            stage=DevelopmentalStage.SENSORIMOTOR,
            name="Sensorimotor",
            description="Acao → Efeito, causalidade basica",
            enabled_modules=['attention', 'perception', 'causal_discovery', 'counterfactual'],
            disabled_modules=['symbolic_abstraction', 'planner', 'meta_learning',
                            'deep_causal', 'high_order_symbolic', 'metacognition',
                            'productive_composition', 'natural_instruction'],
            mastery_threshold=0.7,
            min_experiences=50,
            hyperparameters={'curiosity_weight': 0.3, 'exploration_epsilon': 0.2}
        ),
        DevelopmentalStage.PRE_OPERATIONAL: StageConfig(
            stage=DevelopmentalStage.PRE_OPERATIONAL,
            name="Pre-Operacional",
            description="Representacao simbolica inicial, padroes",
            enabled_modules=['attention', 'perception', 'causal_discovery', 'counterfactual',
                           'symbolic_abstraction'],
            disabled_modules=['planner', 'meta_learning', 'deep_causal',
                            'high_order_symbolic', 'metacognition', 'productive_composition',
                            'natural_instruction'],
            mastery_threshold=0.65,
            min_experiences=100,
            hyperparameters={'curiosity_weight': 0.4, 'exploration_epsilon': 0.15}
        ),
        DevelopmentalStage.CONCRETE_OPERATIONAL: StageConfig(
            stage=DevelopmentalStage.CONCRETE_OPERATIONAL,
            name="Operacional Concreto",
            description="Raciocinio logico, planejamento, composicao",
            enabled_modules=['attention', 'perception', 'causal_discovery', 'counterfactual',
                           'symbolic_abstraction', 'planner', 'productive_composition'],
            disabled_modules=['meta_learning', 'deep_causal', 'high_order_symbolic',
                            'metacognition', 'natural_instruction'],
            mastery_threshold=0.6,
            min_experiences=200,
            hyperparameters={'curiosity_weight': 0.4, 'exploration_epsilon': 0.1}
        ),
        DevelopmentalStage.FORMAL_OPERATIONAL: StageConfig(
            stage=DevelopmentalStage.FORMAL_OPERATIONAL,
            name="Operacional Formal",
            description="Abstracao, causalidade profunda, metacognicao",
            enabled_modules=['attention', 'perception', 'causal_discovery', 'counterfactual',
                           'symbolic_abstraction', 'planner', 'productive_composition',
                           'deep_causal', 'high_order_symbolic', 'metacognition'],
            disabled_modules=['meta_learning', 'natural_instruction'],
            mastery_threshold=0.55,
            min_experiences=500,
            hyperparameters={'curiosity_weight': 0.5, 'exploration_epsilon': 0.08}
        ),
        DevelopmentalStage.META_COGNITIVE: StageConfig(
            stage=DevelopmentalStage.META_COGNITIVE,
            name="Meta-Cognitivo",
            description="Aprender a aprender, adaptacao zero-shot",
            enabled_modules=['attention', 'perception', 'causal_discovery', 'counterfactual',
                           'symbolic_abstraction', 'planner', 'productive_composition',
                           'deep_causal', 'high_order_symbolic', 'metacognition',
                           'meta_learning', 'natural_instruction'],
            disabled_modules=[],
            mastery_threshold=0.5,
            min_experiences=1000,
            hyperparameters={'curiosity_weight': 0.6, 'exploration_epsilon': 0.05}
        )
    }
    
    def __init__(self, 
                 stage_configs: Dict[DevelopmentalStage, StageConfig] = None,
                 initial_stage: DevelopmentalStage = DevelopmentalStage.PRENATAL,
                 allow_regression: bool = False,
                 verbose: bool = True):
        self.stage_configs = stage_configs or self.DEFAULT_STAGES
        self.current_stage = initial_stage
        self.allow_regression = allow_regression
        self.verbose = verbose
        self.current_metrics = DevelopmentalMetrics(stage=initial_stage)
        self.stage_history: List[Dict] = []
        self.on_stage_enter_callbacks: List[Callable] = []
        self.on_stage_exit_callbacks: List[Callable] = []
        self.on_phase_transition_callbacks: List[Callable] = []
        self.transitions: List[Dict] = []
        logger.info(f"DevelopmentalCurriculum initialized at stage: {initial_stage.name}")
    
    def get_enabled_modules(self) -> List[str]:
        config = self.stage_configs[self.current_stage]
        return config.enabled_modules.copy()
    
    def get_disabled_modules(self) -> List[str]:
        config = self.stage_configs[self.current_stage]
        return config.disabled_modules.copy()
    
    def is_module_enabled(self, module_name: str) -> bool:
        config = self.stage_configs[self.current_stage]
        return module_name in config.enabled_modules
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        config = self.stage_configs[self.current_stage]
        return config.hyperparameters.copy()
    
    def update_metrics(self, 
                      success: bool, 
                      error: float = None,
                      module_confidence: Dict[str, float] = None,
                      generalization_score: float = None):
        self.current_metrics.add_experience(success, error)
        if module_confidence:
            for module, conf in module_confidence.items():
                alpha = 0.1
                old = self.current_metrics.module_confidence.get(module, 0)
                self.current_metrics.module_confidence[module] = (1 - alpha) * old + alpha * conf
        if generalization_score is not None:
            alpha = 0.1
            self.current_metrics.generalization_score = (
                (1 - alpha) * self.current_metrics.generalization_score + 
                alpha * generalization_score
            )
    
    def check_stage_transition(self) -> Optional[Tuple[DevelopmentalStage, DevelopmentalStage]]:
        config = self.stage_configs[self.current_stage]
        if self.current_metrics.total_experiences < config.min_experiences:
            return None
        mastery = self.current_metrics.compute_mastery(config.mastery_weights)
        if mastery < config.mastery_threshold:
            return None
        next_stage = self._get_next_stage(self.current_stage)
        if next_stage is None:
            logger.info(f"Estagio final atingido: {self.current_stage.name}")
            return None
        return self._execute_transition(next_stage)
    
    def _get_next_stage(self, current: DevelopmentalStage) -> Optional[DevelopmentalStage]:
        order = list(DevelopmentalStage)
        try:
            idx = order.index(current)
            if idx < len(order) - 1:
                return order[idx + 1]
        except ValueError:
            pass
        return None
    
    def _execute_transition(self, new_stage: DevelopmentalStage) -> Tuple[DevelopmentalStage, DevelopmentalStage]:
        old_stage = self.current_stage
        transition_record = {
            'from_stage': old_stage.name,
            'to_stage': new_stage.name,
            'mastery_at_transition': self.current_metrics.compute_mastery(),
            'total_experiences': self.current_metrics.total_experiences
        }
        self.transitions.append(transition_record)
        self.stage_history.append({
            'stage': old_stage.name,
            'total_experiences': self.current_metrics.total_experiences,
            'success_rate': self.current_metrics.success_rate
        })
        self.current_stage = new_stage
        self.current_metrics = DevelopmentalMetrics(stage=new_stage)
        if self.verbose:
            config = self.stage_configs[new_stage]
            logger.info(f"PHASE TRANSITION: {old_stage.name} -> {new_stage.name}")
        return (old_stage, new_stage)
    
    def apply_hyperparameters(self, agent: Any) -> None:
        config = self.stage_configs[self.current_stage]
        hp = config.hyperparameters
        if hasattr(agent, 'curiosity_module') and 'curiosity_weight' in hp:
            agent.curiosity_module.curiosity_weight = hp['curiosity_weight']
        if hasattr(agent, 'curiosity_module') and 'exploration_epsilon' in hp:
            agent.curiosity_module.exploration_epsilon = hp['exploration_epsilon']
    
    def on_stage_enter(self, callback: Callable):
        self.on_stage_enter_callbacks.append(callback)
    
    def on_stage_exit(self, callback: Callable):
        self.on_stage_exit_callbacks.append(callback)
    
    def on_phase_transition(self, callback: Callable):
        self.on_phase_transition_callbacks.append(callback)
    
    def get_development_report(self) -> Dict:
        return {
            'current_stage': self.current_stage.name,
            'current_metrics': {
                'total_experiences': self.current_metrics.total_experiences,
                'success_rate': self.current_metrics.success_rate,
                'mastery': self.current_metrics.compute_mastery()
            },
            'enabled_modules': self.get_enabled_modules(),
            'disabled_modules': self.get_disabled_modules(),
            'hyperparameters': self.get_hyperparameters(),
            'total_transitions': len(self.transitions)
        }
    
    def get_current_phase_info(self) -> Dict:
        return {
            'phase': self.current_stage.name,
            'description': self.stage_configs[self.current_stage].description,
            'enabled_modules': self.get_enabled_modules(),
            'mastery_threshold': self.stage_configs[self.current_stage].mastery_threshold,
            'episodes_in_phase': self.current_metrics.total_experiences,
            'current_mastery': self.current_metrics.compute_mastery(),
            'progress_to_next': len(self.transitions) / len(self.stage_configs)
        }


# Compatibilidade com codigo antigo
DevelopmentalPhase = DevelopmentalStage
PhaseConfig = StageConfig
