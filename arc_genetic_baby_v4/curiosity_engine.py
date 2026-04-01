"""
curiosity_engine.py

Intrinsic Curiosity Module para ARC-AGI-3 V6

Implementa curiosidade informada como motor de exploracao ativa.
Inspiracao cientifica: Bebes exploram o que e "surpreendente mas compreensivel".

Baseado em:
- Pathak et al. (2017): Curiosity-driven Exploration by Self-supervised Prediction
- Oudeyer & Kaplan (2009): Intrinsic Motivation Systems for Autonomous Mental Development
- Schmidhuber (2010): Formal Theory of Creativity, Fun, and Intrinsic Motivation

Author: ARC-AGI-3 Team
License: MIT
Version: 6.1.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum, auto
import logging
import hashlib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CuriosityType(Enum):
    """Tipos de curiosidade que o sistema pode experimentar"""
    PERCEPTUAL = auto()      # Novidade sensorial (cores, padroes novos)
    STRUCTURAL = auto()      # Novidade relacional (arranjos espaciais)
    CAUSAL = auto()          # Novidade causal (efeitos inesperados)
    CONCEPTUAL = auto()      # Novidade simbolica (regras nao vistas)
    METACOGNITIVE = auto()   # Novidade epistemica (questionar premissas)


@dataclass
class CuriositySignal:
    """
    Sinal de curiosidade computado para uma acao/estado especifico.
    
    Attributes:
        curiosity_value: Valor bruto de curiosidade (0-1)
        curiosity_type: Tipo de curiosidade dominante
        learnability: Quanto potencial de aprendizado existe (0-1)
        novelty: Quao novo e este estimulo (0-1)
        complexity: Complexidade do estimulo (0-1)
        uncertainty: Incerteza do modelo sobre este estimulo (0-1)
        learning_progress: Taxa de melhoria recente (0-1)
        exploration_bonus: Bonus adicional por diversidade de exploracao
    """
    curiosity_value: float
    curiosity_type: CuriosityType
    learnability: float
    novelty: float
    complexity: float
    uncertainty: float
    learning_progress: float
    exploration_bonus: float = 0.0
    
    def __post_init__(self):
        # Garante valores normalizados
        self.curiosity_value = np.clip(self.curiosity_value, 0.0, 1.0)
        self.learnability = np.clip(self.learnability, 0.0, 1.0)
        self.novelty = np.clip(self.novelty, 0.0, 1.0)
        self.complexity = np.clip(self.complexity, 0.0, 1.0)
        self.uncertainty = np.clip(self.uncertainty, 0.0, 1.0)
        self.learning_progress = np.clip(self.learning_progress, -1.0, 1.0)
    
    def to_dict(self) -> Dict:
        """Serializa para logging/debug"""
        return {
            'curiosity_value': self.curiosity_value,
            'curiosity_type': self.curiosity_type.name,
            'learnability': self.learnability,
            'novelty': self.novelty,
            'complexity': self.complexity,
            'uncertainty': self.uncertainty,
            'learning_progress': self.learning_progress,
            'exploration_bonus': self.exploration_bonus
        }


@dataclass
class ExplorationHistory:
    """Historico de exploracao para computar novelty e learning progress"""
    state_hashes: deque = field(default_factory=lambda: deque(maxlen=1000))
    action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    prediction_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    learning_rates: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def add_state(self, state_hash: str):
        self.state_hashes.append(state_hash)
    
    def add_action(self, action: str):
        self.action_counts[action] += 1
    
    def add_prediction_error(self, error: float):
        self.prediction_errors.append(error)
    
    def add_learning_rate(self, rate: float):
        self.learning_rates.append(rate)
    
    def compute_novelty(self, state_hash: str) -> float:
        """Quao novo e este estado comparado ao historico"""
        if not self.state_hashes:
            return 1.0
        return 1.0 - (self.state_hashes.count(state_hash) / len(self.state_hashes))
    
    def compute_action_diversity(self, action: str) -> float:
        """Bonus por explorar acoes menos usadas"""
        if not self.action_counts:
            return 1.0
        total = sum(self.action_counts.values())
        action_freq = self.action_counts.get(action, 0) / total
        return 1.0 - action_freq  # Menos usada = maior bonus
    
    def get_avg_prediction_error(self) -> float:
        return np.mean(self.prediction_errors) if self.prediction_errors else 0.0
    
    def get_avg_learning_rate(self) -> float:
        return np.mean(self.learning_rates) if self.learning_rates else 0.0


class WorldModelPredictor(ABC):
    """Interface para modelo de mundo que faz predicoes"""
    
    @abstractmethod
    def predict(self, state: np.ndarray, action: str) -> Tuple[np.ndarray, float]:
        """
        Prediz proximo estado dado estado atual e acao.
        
        Returns:
            (predicted_state, uncertainty)
        """
        pass
    
    @abstractmethod
    def compute_prediction_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Computa erro entre predicao e realidade"""
        pass


class IntrinsicCuriosityModule:
    """
    Modulo principal de Curiosidade Intrinseca para ARC-AGI-3 V6.
    
    Computa recompensa intrinseca baseada em:
    1. Novidade do estado (nunca visto antes)
    2. Surpresa (erro de predicao do modelo de mundo)
    3. Aprendibilidade (quao "compreensivel" e a surpresa)
    4. Progresso no aprendizado (estou melhorando?)
    5. Diversidade de exploracao (estou explorando acoes variadas?)
    
    Formula principal:
        Curiosity = Novelty x Learnability x (1 + LearningProgress) x DiversityBonus
    
    Onde:
        Learnability = PredictionError x (1 - Uncertainty)
        -> Alta surpresa + baixa incerteza = "posso aprender isso!"
        -> Alta surpresa + alta incerteza = "nao entendo nada" -> evitar
    """
    
    def __init__(self, 
                 world_model: WorldModelPredictor = None,
                 novelty_decay: float = 0.99,
                 curiosity_weight: float = 0.4,
                 exploration_epsilon: float = 0.1,
                 min_curiosity_threshold: float = 0.1):
        """
        Args:
            world_model: Modelo de mundo para predicoes (pode ser counterfactual.py)
            novelty_decay: Taxa de decaimento da novidade (menor = esquece mais rapido)
            curiosity_weight: Peso da curiosidade na selecao de acoes (0-1)
            exploration_epsilon: Probabilidade de exploracao aleatoria
            min_curiosity_threshold: Minimo de curiosidade para considerar acao
        """
        self.world_model = world_model
        self.novelty_decay = novelty_decay
        self.curiosity_weight = curiosity_weight
        self.exploration_epsilon = exploration_epsilon
        self.min_curiosity_threshold = min_curiosity_threshold
        
        # Historico para computar novelty e learning progress
        self.history = ExplorationHistory()
        
        # Mapa de novidade por tipo de feature
        self.feature_novelty: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Contadores para estatisticas
        self.total_explorations = 0
        self.curiosity_driven_actions = 0
        
        # Cache de hashes de estado
        self._state_cache: Dict[int, str] = {}
    
    def compute_state_hash(self, state: np.ndarray) -> str:
        """Gera hash unico para um estado (para tracking de novelty)"""
        # Usa cache para eficiencia
        state_id = id(state)
        if state_id not in self._state_cache:
            # Hash baseado em features principais (nao pixels brutos)
            features = self._extract_state_features(state)
            self._state_cache[state_id] = hashlib.md5(
                str(features).encode()
            ).hexdigest()[:16]
        
        # Limpa cache periodicamente
        if len(self._state_cache) > 1000:
            self._state_cache.clear()
        
        return self._state_cache[state_id]
    
    def _extract_state_features(self, state: np.ndarray) -> Dict[str, Any]:
        """Extrai features de alto nivel para hashing (mais robusto que pixels)"""
        return {
            'shape': state.shape,
            'n_colors': len(np.unique(state)),
            'n_objects': self._count_objects(state),
            'symmetry': self._detect_symmetry(state),
            'dominant_color': int(np.bincount(state.flatten()).argmax()),
            'complexity': self._estimate_complexity(state)
        }
    
    def _count_objects(self, grid: np.ndarray) -> int:
        """Conta objetos conectados (conectividade 4)"""
        from scipy import ndimage
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
        
        total = 0
        for color in range(16):
            mask = (grid == color)
            if np.any(mask):
                _, num = ndimage.label(mask.astype(int), structure=structure)
                total += num
        
        return total
    
    def _detect_symmetry(self, grid: np.ndarray) -> int:
        """Detecta tipo de simetria (0=none, 1=horizontal, 2=vertical, 3=both)"""
        h_sym = np.array_equal(grid, np.fliplr(grid))
        v_sym = np.array_equal(grid, np.flipud(grid))
        
        if h_sym and v_sym:
            return 3
        elif h_sym:
            return 1
        elif v_sym:
            return 2
        return 0
    
    def _estimate_complexity(self, grid: np.ndarray) -> float:
        """Estima complexidade visual do grid"""
        n_colors = len(np.unique(grid))
        n_objects = self._count_objects(grid)
        entropy = self._compute_entropy(grid)
        
        # Normaliza para 0-1
        complexity = (n_colors / 16 + n_objects / 50 + entropy) / 3
        return np.clip(complexity, 0, 1)
    
    def _compute_entropy(self, grid: np.ndarray) -> float:
        """Computa entropia de Shannon da distribuicao de cores"""
        counts = np.bincount(grid.flatten())
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log2(probs)) / np.log2(16)  # Normalizado
    
    def compute_curiosity_signal(self, 
                                 state: np.ndarray,
                                 action: str,
                                 predicted_state: np.ndarray = None,
                                 actual_state: np.ndarray = None) -> CuriositySignal:
        """
        Computa sinal completo de curiosidade para um par (estado, acao).
        
        Args:
            state: Estado atual
            action: Acao sendo considerada
            predicted_state: Estado predito pelo modelo de mundo (opcional)
            actual_state: Estado real observado (opcional, para learning progress)
        
        Returns:
            CuriositySignal com todos os componentes
        """
        # 1. COMPUTA NOVIDADE
        state_hash = self.compute_state_hash(state)
        novelty = self.history.compute_novelty(state_hash)
        
        # Atualiza historico
        self.history.add_state(state_hash)
        self.history.add_action(action)
        
        # 2. COMPUTA INCERTEZA DO MODELO
        if self.world_model and predicted_state is not None:
            _, uncertainty = self.world_model.predict(state, action)
        else:
            # Fallback: usa complexidade como proxy para incerteza
            uncertainty = self._estimate_complexity(state)
        
        # 3. COMPUTA ERRO DE PREDICAO (SURPRESA)
        if actual_state is not None and predicted_state is not None:
            prediction_error = self._compute_prediction_error(predicted_state, actual_state)
            self.history.add_prediction_error(prediction_error)
        else:
            prediction_error = self.history.get_avg_prediction_error()
        
        # 4. COMPUTA APRENDIBILIDADE
        # Learnability = Surpresa x (1 - Incerteza)
        # -> Queremos surpresa "compreensivel", nao caos total
        learnability = prediction_error * (1 - uncertainty)
        
        # 5. COMPUTA PROGRESSO NO APRENDIZADO
        # Se erro de predicao esta diminuindo -> estamos aprendendo
        if len(self.history.prediction_errors) >= 5:
            recent_errors = list(self.history.prediction_errors)[-5:]
            older_errors = list(self.history.prediction_errors)[-10:-5] if len(self.history.prediction_errors) >= 10 else recent_errors
            
            if np.mean(older_errors) > 0:
                learning_progress = 1.0 - (np.mean(recent_errors) / np.mean(older_errors))
            else:
                learning_progress = 0.0
        else:
            learning_progress = 0.0
        
        self.history.add_learning_rate(learning_progress)
        
        # 6. COMPUTA BONUS DE DIVERSIDADE
        diversity_bonus = self.history.compute_action_diversity(action)
        
        # 7. COMPUTA COMPLEXIDADE DO ESTIMULO
        complexity = self._estimate_complexity(state)
        
        # 8. FORMULA FINAL DE CURIOSIDADE
        # Curiosidade = Novidade x Aprendibilidade x (1 + Progresso) x Diversidade
        curiosity_value = (
            novelty * 
            learnability * 
            (1.0 + np.clip(learning_progress, 0, 1)) * 
            (1.0 + 0.3 * diversity_bonus)
        )
        
        # Normaliza para 0-1 (aproximado)
        curiosity_value = np.clip(curiosity_value / 0.5, 0, 1)  # Divide por valor tipico maximo
        
        # 9. DETERMINA TIPO DE CURIOSIDADE DOMINANTE
        curiosity_type = self._determine_curiosity_type(state, action, novelty, prediction_error)
        
        # 10. APLICA EXPLORATION BONUS
        exploration_bonus = 0.0
        if np.random.random() < self.exploration_epsilon:
            exploration_bonus = 0.2  # Bonus fixo para exploracao aleatoria
        
        signal = CuriositySignal(
            curiosity_value=curiosity_value,
            curiosity_type=curiosity_type,
            learnability=learnability,
            novelty=novelty,
            complexity=complexity,
            uncertainty=uncertainty,
            learning_progress=learning_progress,
            exploration_bonus=exploration_bonus
        )
        
        # Atualiza contadores
        self.total_explorations += 1
        if curiosity_value > self.min_curiosity_threshold:
            self.curiosity_driven_actions += 1
        
        return signal
    
    def _compute_prediction_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Computa erro normalizado entre predicao e realidade"""
        if predicted.shape != actual.shape:
            return 1.0  # Erro maximo se shapes diferentes
        
        # MSE normalizado por dominio (0-15 para ARC)
        mse = np.mean((predicted.astype(float) - actual.astype(float)) ** 2)
        normalized_mse = mse / (15 ** 2)  # Normaliza para 0-1
        
        return np.clip(normalized_mse, 0, 1)
    
    def _determine_curiosity_type(self, 
                                  state: np.ndarray,
                                  action: str,
                                  novelty: float,
                                  prediction_error: float) -> CuriosityType:
        """Determina qual tipo de curiosidade esta dominando"""
        # Heuristicas baseadas em features do estado
        
        # Alta novidade + baixo erro = Perceptual
        if novelty > 0.7 and prediction_error < 0.3:
            return CuriosityType.PERCEPTUAL
        
        # Alto erro + features estruturais = Structural
        if prediction_error > 0.5:
            if self._detect_symmetry(state) > 0:
                return CuriosityType.STRUCTURAL
            return CuriosityType.CAUSAL
        
        # Complexidade alta = Conceptual
        if self._estimate_complexity(state) > 0.7:
            return CuriosityType.CONCEPTUAL
        
        # Default
        return CuriosityType.CAUSAL
    
    def select_action_with_curiosity(self,
                                     state: np.ndarray,
                                     available_actions: List[str],
                                     base_action_scores: Dict[str, float] = None,
                                     world_model = None) -> Tuple[str, Dict]:
        """
        Seleciona acao combinando score base (ex: Free Energy) + curiosidade.
        
        Args:
            state: Estado atual
            available_actions: Lista de acoes possiveis
            base_action_scores: Scores de outro sistema (ex: Active Inference)
            world_model: Modelo de mundo para predicoes (override opcional)
        
        Returns:
            (acao_escolhida, info_dict com detalhes da decisao)
        """
        if not available_actions:
            raise ValueError("Nenhuma acao disponivel")
        
        # Se exploration epsilon -> acao aleatoria
        if np.random.random() < self.exploration_epsilon:
            action = np.random.choice(available_actions)
            return action, {
                'selection_method': 'random_exploration',
                'epsilon': self.exploration_epsilon
            }
        
        # Computa curiosidade para cada acao
        action_curiosities = {}
        world_model = world_model or self.world_model
        
        for action in available_actions:
            # Prediz consequencia
            if world_model:
                predicted_state, uncertainty = world_model.predict(state, action)
            else:
                predicted_state = state  # Fallback
                uncertainty = 0.5
            
            # Computa sinal de curiosidade
            signal = self.compute_curiosity_signal(
                state=state,
                action=action,
                predicted_state=predicted_state
            )
            
            # Combina com score base se disponivel
            base_score = base_action_scores.get(action, 0.5) if base_action_scores else 0.5
            
            # Score combinado: (1-w)*base + w*curiosity
            combined_score = (
                (1 - self.curiosity_weight) * base_score + 
                self.curiosity_weight * (signal.curiosity_value + signal.exploration_bonus)
            )
            
            action_curiosities[action] = {
                'combined_score': combined_score,
                'base_score': base_score,
                'curiosity_signal': signal
            }
        
        # Seleciona acao com maior score combinado
        best_action = max(action_curiosities.keys(), 
                         key=lambda a: action_curiosities[a]['combined_score'])
        
        info = {
            'selection_method': 'curiosity_weighted',
            'selected_action': best_action,
            'selected_score': action_curiosities[best_action]['combined_score'],
            'curiosity_weight': self.curiosity_weight,
            'all_actions': {
                a: {
                    'score': v['combined_score'],
                    'curiosity': v['curiosity_signal'].curiosity_value,
                    'novelty': v['curiosity_signal'].novelty,
                    'learnability': v['curiosity_signal'].learnability
                }
                for a, v in action_curiosities.items()
            }
        }
        
        return best_action, info
    
    def update_from_outcome(self, 
                           state: np.ndarray,
                           action: str,
                           actual_next_state: np.ndarray,
                           predicted_next_state: np.ndarray = None,
                           learning_rate: float = None):
        """
        Atualiza modelo de curiosidade baseado no resultado real de uma acao.
        
        Args:
            state: Estado antes da acao
            action: Acao executada
            actual_next_state: Estado real apos acao
            predicted_next_state: Estado que foi predito (para computar erro)
            learning_rate: Taxa de aprendizado observada (opcional)
        """
        # Computa erro real de predicao
        if predicted_next_state is not None:
            prediction_error = self._compute_prediction_error(
                predicted_next_state, actual_next_state
            )
            self.history.add_prediction_error(prediction_error)
        
        # Atualiza learning rate se fornecido
        if learning_rate is not None:
            self.history.add_learning_rate(learning_rate)
        
        # Decai novidade de estados visitados
        state_hash = self.compute_state_hash(state)
        # (Novelty ja e computada como frequencia no historico)
        
        # Logging
        logger.debug(
            f"Curiosity update: action={action}, "
            f"prediction_error={prediction_error if predicted_next_state else 'N/A'}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatisticas sobre comportamento de exploracao"""
        return {
            'total_explorations': self.total_explorations,
            'curiosity_driven_actions': self.curiosity_driven_actions,
            'curiosity_driven_ratio': (
                self.curiosity_driven_actions / max(1, self.total_explorations)
            ),
            'avg_novelty': np.mean([
                self.history.compute_novelty(h) for h in list(self.history.state_hashes)[-100:]
            ]) if self.history.state_hashes else 0.0,
            'avg_prediction_error': self.history.get_avg_prediction_error(),
            'avg_learning_progress': self.history.get_avg_learning_rate(),
            'action_distribution': dict(self.history.action_counts),
            'curiosity_weight': self.curiosity_weight,
            'exploration_epsilon': self.exploration_epsilon
        }
    
    def reset(self, keep_history: bool = False):
        """Reseta estado interno (para novo episodio)"""
        if not keep_history:
            self.history = ExplorationHistory()
            self.feature_novelty.clear()
        
        self._state_cache.clear()
    
    def save_checkpoint(self, path: str):
        """Salva estado para checkpointing"""
        import json
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'total_explorations': self.total_explorations,
            'curiosity_driven_actions': self.curiosity_driven_actions,
            'history': {
                'state_hashes': list(self.history.state_hashes),
                'action_counts': dict(self.history.action_counts),
                'prediction_errors': list(self.history.prediction_errors),
                'learning_rates': list(self.history.learning_rates)
            },
            'config': {
                'novelty_decay': self.novelty_decay,
                'curiosity_weight': self.curiosity_weight,
                'exploration_epsilon': self.exploration_epsilon,
                'min_curiosity_threshold': self.min_curiosity_threshold
            }
        }
        
        with open(path / 'curiosity_module.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, path: str) -> 'IntrinsicCuriosityModule':
        """Carrega estado de checkpoint"""
        import json
        from pathlib import Path
        
        path = Path(path)
        
        with open(path / 'curiosity_module.json', 'r') as f:
            checkpoint = json.load(f)
        
        module = cls(
            novelty_decay=checkpoint['config']['novelty_decay'],
            curiosity_weight=checkpoint['config']['curiosity_weight'],
            exploration_epsilon=checkpoint['config']['exploration_epsilon'],
            min_curiosity_threshold=checkpoint['config']['min_curiosity_threshold']
        )
        
        module.total_explorations = checkpoint['total_explorations']
        module.curiosity_driven_actions = checkpoint['curiosity_driven_actions']
        
        # Restaura historico
        module.history.state_hashes = deque(
            checkpoint['history']['state_hashes'], 
            maxlen=1000
        )
        module.history.action_counts = defaultdict(
            int, checkpoint['history']['action_counts']
        )
        module.history.prediction_errors = deque(
            checkpoint['history']['prediction_errors'],
            maxlen=100
        )
        module.history.learning_rates = deque(
            checkpoint['history']['learning_rates'],
            maxlen=50
        )
        
        return module
