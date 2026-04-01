"""
self_play_engine.py

Self-Play Data Generation Engine para ARC-AGI-3 V6

Implementa geracao automatica de dados de treino via auto-interacao.
Inspiracao: Bebes nao esperam exemplos prontos - criam seus proprios dados
agindo no mundo, observando consequencias, e repetindo com variacoes.

Based on:
- Silver et al. (2017): Mastering Chess and Shogi by Self-Play
- Bansal et al. (2018): Emergent Complexity via Multi-Agent Competition

Version: 6.3.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class PuzzleType(Enum):
    """Tipos de puzzles ARC que podem ser gerados"""
    SYMMETRY = auto()
    ROTATION = auto()
    COLOR_MAPPING = auto()
    OBJECT_COUNTING = auto()
    PATTERN_COMPLETION = auto()
    GRAVITY = auto()
    FILL_HOLES = auto()
    OBJECT_MOVEMENT = auto()
    SCALING = auto()
    NOISE_REMOVAL = auto()
    COMPOSITIONAL = auto()


@dataclass
class TrainingExample:
    """Exemplo de treino gerado por self-play"""
    input_grid: np.ndarray
    target_grid: np.ndarray
    action: str
    action_params: Dict[str, Any]
    puzzle_type: PuzzleType
    difficulty: float
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SelfPlaySession:
    """Registro de uma sessao de self-play"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    n_examples_generated: int = 0
    n_examples_learned: int = 0
    avg_difficulty: float = 0.0
    avg_reward: float = 0.0
    puzzle_type_distribution: Dict[str, int] = field(default_factory=dict)


class TransformationPrimitive(ABC):
    """Classe base para primitivas de transformacao de grid"""
    
    @abstractmethod
    def apply(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        pass
    
    @abstractmethod
    def inverse(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class RotatePrimitive(TransformationPrimitive):
    def apply(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        k = params.get('k', 1) if params else 1
        return np.rot90(grid, k=k)
    
    def inverse(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        k = params.get('k', 1) if params else 1
        return np.rot90(grid, k=-k)
    
    def get_name(self) -> str:
        return 'rotate'


class FlipPrimitive(TransformationPrimitive):
    def apply(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        axis = params.get('axis', 0) if params else 0
        return np.flip(grid, axis=axis)
    
    def inverse(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        return self.apply(grid, params)
    
    def get_name(self) -> str:
        return 'flip'


class ColorMapPrimitive(TransformationPrimitive):
    def apply(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        color_map = params.get('color_map', {}) if params else {}
        result = grid.copy()
        for src, dst in color_map.items():
            result[grid == src] = dst
        return result
    
    def inverse(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        color_map = params.get('color_map', {}) if params else {}
        inverse_map = {v: k for k, v in color_map.items()}
        result = grid.copy()
        for src, dst in inverse_map.items():
            result[grid == src] = dst
        return result
    
    def get_name(self) -> str:
        return 'color_map'


class TranslatePrimitive(TransformationPrimitive):
    def apply(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        dx = params.get('dx', 0) if params else 0
        dy = params.get('dy', 0) if params else 0
        return self._safe_translate(grid, dx, dy)
    
    def _safe_translate(self, grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        result = np.zeros_like(grid)
        h, w = grid.shape
        for y in range(h):
            for x in range(w):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    result[ny, nx] = grid[y, x]
        return result
    
    def inverse(self, grid: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        dx = params.get('dx', 0) if params else 0
        dy = params.get('dy', 0) if params else 0
        return self._safe_translate(grid, -dx, -dy)
    
    def get_name(self) -> str:
        return 'translate'


class SelfPlayEngine:
    """
    Engine principal de Self-Play para ARC-AGI-3 V6.
    
    Gera dados de treino infinitos e personalizados atraves de:
    1. Transformacoes aleatorias em grids base
    2. Criacao de pares (input, target) via transformacoes inversas
    3. Curriculum de dificuldade adaptativa
    4. Replay de exemplos dificeis
    """
    
    PRIMITIVES = {
        'rotate': RotatePrimitive(),
        'flip': FlipPrimitive(),
        'color_map': ColorMapPrimitive(),
        'translate': TranslatePrimitive(),
    }
    
    PUZZLE_PRIMITIVES = {
        PuzzleType.SYMMETRY: ['flip'],
        PuzzleType.ROTATION: ['rotate'],
        PuzzleType.COLOR_MAPPING: ['color_map'],
        PuzzleType.OBJECT_MOVEMENT: ['translate'],
        PuzzleType.COMPOSITIONAL: ['rotate', 'flip', 'color_map', 'translate'],
    }
    
    def __init__(self,
                 grid_size: int = 10,
                 num_colors: int = 10,
                 difficulty_adaptation_rate: float = 0.1,
                 replay_buffer_size: int = 1000,
                 min_difficulty: float = 0.1,
                 max_difficulty: float = 0.9,
                 target_success_rate: float = 0.7):
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.difficulty_adaptation_rate = difficulty_adaptation_rate
        self.replay_buffer_size = replay_buffer_size
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.target_success_rate = target_success_rate
        
        self.replay_buffer: deque = deque(maxlen=replay_buffer_size)
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.current_difficulty: Dict[PuzzleType, float] = {pt: 0.3 for pt in PuzzleType}
        self.current_session: Optional[SelfPlaySession] = None
        self.session_history: List[SelfPlaySession] = []
        self.total_examples_generated = 0
        self.total_examples_learned = 0

    def start_session(self, session_id: str = None) -> SelfPlaySession:
        if session_id is None:
            import hashlib
            session_id = hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:12]
        self.current_session = SelfPlaySession(session_id=session_id, start_time=datetime.now())
        logger.info(f"Self-play session started: {session_id}")
        return self.current_session

    def end_session(self) -> Optional[SelfPlaySession]:
        if self.current_session is None:
            return None
        self.current_session.end_time = datetime.now()
        self.session_history.append(self.current_session)
        session = self.current_session
        self.current_session = None
        logger.info(f"Self-play session ended: {session.session_id}, {session.n_examples_generated} examples")
        return session

    def generate_base_grid(self, puzzle_type: PuzzleType = None, complexity: float = 0.5) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        n_objects = int(1 + complexity * 5)
        
        for _ in range(n_objects):
            color = np.random.randint(1, self.num_colors)
            obj_size = np.random.randint(1, int(3 + complexity * 3))
            max_x = self.grid_size - obj_size
            max_y = self.grid_size - obj_size
            if max_x <= 0 or max_y <= 0:
                continue
            x = np.random.randint(0, max_x + 1)
            y = np.random.randint(0, max_y + 1)
            for dy in range(obj_size):
                for dx in range(obj_size):
                    if y + dy < self.grid_size and x + dx < self.grid_size:
                        grid[y + dy, x + dx] = color
        
        if puzzle_type == PuzzleType.SYMMETRY:
            grid = self._make_symmetric(grid)
        return grid

    def _make_symmetric(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        result = grid.copy()
        for y in range(h):
            for x in range(w // 2):
                result[y, w - 1 - x] = grid[y, x]
        return result

    def generate_example(self, puzzle_type: PuzzleType = None, difficulty: float = None, use_replay: bool = False) -> Optional[TrainingExample]:
        if use_replay and len(self.replay_buffer) > 0:
            if np.random.random() < 0.3:
                return random.choice(list(self.replay_buffer))
        
        if puzzle_type is None:
            puzzle_type = np.random.choice(list(PuzzleType))
        
        if difficulty is None:
            difficulty = self.current_difficulty.get(puzzle_type, 0.5)
        
        base_grid = self.generate_base_grid(puzzle_type, difficulty)
        available_primitives = self.PUZZLE_PRIMITIVES.get(puzzle_type, list(self.PRIMITIVES.keys()))
        
        n_transforms = int(1 + difficulty * 3)
        input_grid = base_grid.copy()
        action_sequence = []
        
        for _ in range(n_transforms):
            primitive_name = np.random.choice(available_primitives)
            primitive = self.PRIMITIVES[primitive_name]
            params = self._generate_primitive_params(primitive_name, difficulty)
            input_grid = primitive.apply(input_grid, params)
            action_sequence.append((primitive_name, params))
        
        target_grid = base_grid
        reward = self._compute_intrinsic_reward(difficulty, puzzle_type)
        
        example = TrainingExample(
            input_grid=input_grid,
            target_grid=target_grid,
            action=action_sequence[-1][0] if action_sequence else 'identity',
            action_params=action_sequence[-1][1] if action_sequence else {},
            puzzle_type=puzzle_type,
            difficulty=difficulty,
            reward=reward,
            metadata={'action_sequence': action_sequence, 'n_transforms': n_transforms}
        )
        
        self.total_examples_generated += 1
        if self.current_session:
            self.current_session.n_examples_generated += 1
        
        return example

    def _generate_primitive_params(self, primitive_name: str, difficulty: float) -> Dict[str, Any]:
        params = {}
        if primitive_name == 'rotate':
            params['k'] = np.random.choice([1, 2, 3])
        elif primitive_name == 'flip':
            params['axis'] = np.random.choice([0, 1])
        elif primitive_name == 'color_map':
            n_colors_to_map = int(1 + difficulty * 3)
            colors = np.random.choice(range(1, self.num_colors), size=min(n_colors_to_map, self.num_colors - 1), replace=False)
            params['color_map'] = {int(c): int(np.random.randint(1, self.num_colors)) for c in colors}
        elif primitive_name == 'translate':
            max_shift = int(1 + difficulty * 3)
            params['dx'] = np.random.randint(-max_shift, max_shift + 1)
            params['dy'] = np.random.randint(-max_shift, max_shift + 1)
        return params

    def _compute_intrinsic_reward(self, difficulty: float, puzzle_type: PuzzleType) -> float:
        history = self.performance_history[puzzle_type]
        success_rate = np.mean(history) if len(history) > 0 else 0.5
        target_diff = abs(success_rate - self.target_success_rate)
        base_reward = 1.0 - target_diff
        difficulty_bonus = 1.0 + difficulty * 0.5
        return np.clip(base_reward * difficulty_bonus, 0, 2)

    def update_from_outcome(self, example: TrainingExample, success: bool, learning_progress: float = None):
        self.performance_history[example.puzzle_type].append(1 if success else 0)
        
        if success:
            self.total_examples_learned += 1
            if self.current_session:
                self.current_session.n_examples_learned += 1
        
        current_diff = self.current_difficulty[example.puzzle_type]
        if success:
            new_diff = current_diff + self.difficulty_adaptation_rate * (1 - current_diff)
        else:
            new_diff = current_diff - self.difficulty_adaptation_rate * current_diff
        
        new_diff = np.clip(new_diff, self.min_difficulty, self.max_difficulty)
        self.current_difficulty[example.puzzle_type] = new_diff
        
        if not success and example.difficulty > 0.5:
            self.replay_buffer.append(example)

    def generate_curriculum_batch(self, batch_size: int = 32, focus_type: PuzzleType = None) -> List[TrainingExample]:
        examples = []
        for _ in range(batch_size):
            use_replay = np.random.random() < 0.2
            example = self.generate_example(puzzle_type=focus_type, use_replay=use_replay)
            if example:
                examples.append(example)
        return examples

    def get_statistics(self) -> Dict[str, Any]:
        performance = {}
        for puzzle_type in PuzzleType:
            history = self.performance_history[puzzle_type]
            if len(history) > 0:
                performance[puzzle_type.name] = {
                    'success_rate': float(np.mean(history)),
                    'n_samples': len(history),
                    'current_difficulty': float(self.current_difficulty[puzzle_type])
                }
        
        return {
            'total_examples_generated': self.total_examples_generated,
            'total_examples_learned': self.total_examples_learned,
            'learning_rate': self.total_examples_learned / max(1, self.total_examples_generated),
            'replay_buffer_size': len(self.replay_buffer),
            'performance_by_type': performance,
            'current_difficulty': {pt.name: float(diff) for pt, diff in self.current_difficulty.items()},
            'sessions_completed': len(self.session_history)
        }

    def get_generation_stats(self) -> Dict[str, Any]:
        """Alias para compatibilidade com codigo antigo"""
        return self.get_statistics()


# Compatibilidade com codigo antigo
SelfPlayDataGenerator = SelfPlayEngine


class SelfPlayDataGenerator:
    """
    Gera dados de treino autogerados via interacao com ambiente simulado.
    
    Implementa estrategias de:
    - Exploracao curiosidade-dirigida
    - Curriculo adaptativo de dificuldade
    - Auto-avaliacao de qualidade
    """
    
    # Transformacoes disponiveis para self-play
    TRANSFORMATIONS = {
        'rotate_90': lambda grid: np.rot90(grid, k=1),
        'rotate_180': lambda grid: np.rot90(grid, k=2),
        'rotate_270': lambda grid: np.rot90(grid, k=3),
        'flip_h': lambda grid: np.fliplr(grid),
        'flip_v': lambda grid: np.flipud(grid),
        'color_shift': lambda grid: (grid + 1) % 10,  # Assumindo 10 cores
        'color_invert': lambda grid: 9 - grid,
        'identity': lambda grid: grid.copy(),
        'swap_quadrants': lambda grid: _swap_quadrants(grid),
        'mirror_diag': lambda grid: np.transpose(grid)
    }
    
    def __init__(self,
                 skill_estimator: Callable = None,
                 curiosity_module = None,
                 min_difficulty: float = 0.1,
                 max_difficulty: float = 1.0):
        """
        Args:
            skill_estimator: Funcao que estima nivel de habilidade do agente (0-1)
            curiosity_module: Modulo de curiosidade para guiar exploracao
            min_difficulty: Dificuldade minima de puzzles gerados
            max_difficulty: Dificuldade maxima de puzzles gerados
        """
        self.skill_estimator = skill_estimator or (lambda: 0.5)
        self.curiosity_module = curiosity_module
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        
        # Historico de exemplos gerados
        self.generated_examples: deque = deque(maxlen=1000)
        self.generation_stats: Dict[str, int] = {
            'total_generated': 0,
            'by_difficulty': {'easy': 0, 'medium': 0, 'hard': 0},
            'by_transformation': {k: 0 for k in self.TRANSFORMATIONS.keys()}
        }
        
        # Taxa de sucesso por tipo de transformacao
        self.success_rates: Dict[str, List[float]] = {
            k: [0.5] for k in self.TRANSFORMATIONS.keys()
        }
        
        logger.info("SelfPlayDataGenerator inicializado")
    
    def generate_episode(self, 
                        base_grid: Optional[np.ndarray] = None,
                        n_steps: int = 10,
                        skill_level: Optional[float] = None) -> List[TrainingExample]:
        """
        Cria sequencia de exemplos via auto-interacao.
        
        Args:
            base_grid: Grid base inicial (opcional)
            n_steps: Numero de passos na sequencia
            skill_level: Nivel de habilidade do agente (auto-detectado se None)
        
        Returns:
            Lista de TrainingExample
        """
        if skill_level is None:
            skill_level = self.skill_estimator()
        
        # Gera grid base se nao fornecido
        if base_grid is None:
            base_grid = self._generate_base_grid(skill_level)
        
        examples = []
        current_grid = base_grid.copy()
        
        for step in range(n_steps):
            # Seleciona transformacao baseada em curiosidade ou dificuldade
            action = self._select_exploratory_action(current_grid, skill_level)
            
            # Aplica transformacao
            transformed_grid = self._apply_transformation(current_grid, action)
            
            # Computa recompensa intrinseca (quanto aprende com isso?)
            intrinsic_reward = self._compute_intrinsic_reward(
                current_grid, transformed_grid, action, skill_level
            )
            
            # Armazena exemplo
            example = TrainingExample(
                input_grid=transformed_grid,
                target_grid=current_grid,  # Objetivo: reverter a transformacao
                action=action,
                reward=intrinsic_reward,
                metadata={
                    'self_generated': True,
                    'step': step,
                    'skill_level': skill_level,
                    'generation_method': 'self_play'
                },
                difficulty=self._estimate_difficulty(action, current_grid)
            )
            
            examples.append(example)
            self.generated_examples.append(example)
            self._update_stats(example)
            
            # Atualiza estado com probabilidade de continuar sequencia
            if random.random() < 0.3:
                current_grid = transformed_grid
        
        return examples
    
    def _generate_base_grid(self, skill_level: float) -> np.ndarray:
        """Gera grid base adaptado ao nivel de habilidade"""
        # Tamanho adaptativo
        if skill_level < 0.3:
            size = random.randint(5, 10)
        elif skill_level < 0.6:
            size = random.randint(10, 20)
        else:
            size = random.randint(20, 30)
        
        # Gera grid com estrutura (nao completamente aleatorio)
        grid = np.zeros((size, size), dtype=int)
        
        # Adiciona alguns objetos/padroes
        n_objects = random.randint(1, max(1, int(5 * skill_level)))
        
        for _ in range(n_objects):
            # Tamanho do objeto
            obj_size = random.randint(2, max(2, int(size * 0.2)))
            
            # Posicao
            x = random.randint(0, size - obj_size)
            y = random.randint(0, size - obj_size)
            
            # Cor
            color = random.randint(1, 9)
            
            # Forma (retangulo ou quadrado)
            grid[x:x+obj_size, y:y+obj_size] = color
        
        return grid
    
    def _select_exploratory_action(self, 
                                   current_grid: np.ndarray, 
                                   skill_level: float) -> str:
        """Seleciona acao para "brincar" (explorar espaco de transformacoes)"""
        available = list(self.TRANSFORMATIONS.keys())
        
        # Se tem curiosidade, usa ela
        if self.curiosity_module:
            # Computa curiosidade para cada transformacao
            curiosity_scores = []
            for action in available:
                signal = self.curiosity_module.compute_curiosity_signal(
                    state=current_grid,
                    action=action
                )
                curiosity_scores.append(signal.curiosity_value)
            
            # 80% das vezes: escolhe por curiosidade
            # 20% das vezes: escolhe por dificuldade apropriada
            if random.random() < 0.8:
                return available[np.argmax(curiosity_scores)]
        
        # Seleciona por dificuldade apropriada ao skill level
        # Transformacoes mais complexas para niveis mais altos
        if skill_level < 0.3:
            easy_actions = ['identity', 'color_shift', 'flip_h', 'flip_v']
            return random.choice([a for a in easy_actions if a in available])
        elif skill_level < 0.6:
            medium_actions = ['rotate_90', 'rotate_180', 'flip_h', 'color_shift']
            return random.choice([a for a in medium_actions if a in available])
        else:
            hard_actions = ['rotate_90', 'rotate_180', 'rotate_270', 
                          'swap_quadrants', 'mirror_diag']
            return random.choice([a for a in hard_actions if a in available])
    
    def _apply_transformation(self, 
                             grid: np.ndarray, 
                             action: str) -> np.ndarray:
        """Aplica transformacao ao grid"""
        if action in self.TRANSFORMATIONS:
            try:
                return self.TRANSFORMATIONS[action](grid)
            except Exception as e:
                logger.warning(f"Erro aplicando transformacao {action}: {e}")
                return grid.copy()
        return grid.copy()
    
    def _compute_intrinsic_reward(self,
                                  original: np.ndarray,
                                  transformed: np.ndarray,
                                  action: str,
                                  skill_level: float) -> float:
        """
        Computa recompensa intrinseca baseada em:
        - Diferenca entre grids (novidade)
        - Apropriada ao nivel de habilidade (curriculo)
        - Diversidade de transformacoes
        """
        # Recompensa base pela diferenca
        diff = np.mean(original != transformed)
        base_reward = diff * 0.5  # 0-0.5
        
        # Bonus por dificuldade apropriada
        action_difficulty = self._estimate_action_difficulty(action)
        difficulty_match = 1.0 - abs(action_difficulty - skill_level)
        difficulty_bonus = difficulty_match * 0.3  # 0-0.3
        
        # Bonus por diversidade (menos usada = mais bonus)
        usage_count = self.generation_stats['by_transformation'].get(action, 0)
        total = self.generation_stats['total_generated'] + 1
        diversity_bonus = (1.0 - usage_count / total) * 0.2  # 0-0.2
        
        total_reward = base_reward + difficulty_bonus + diversity_bonus
        return np.clip(total_reward, 0, 1)
    
    def _estimate_action_difficulty(self, action: str) -> float:
        """Estima dificuldade de uma transformacao (0-1)"""
        difficulties = {
            'identity': 0.1,
            'flip_h': 0.2,
            'flip_v': 0.2,
            'color_shift': 0.3,
            'color_invert': 0.3,
            'rotate_90': 0.5,
            'rotate_270': 0.5,
            'rotate_180': 0.6,
            'mirror_diag': 0.7,
            'swap_quadrants': 0.8
        }
        return difficulties.get(action, 0.5)
    
    def _estimate_difficulty(self, action: str, grid: np.ndarray) -> float:
        """Estima dificuldade total do exemplo"""
        action_diff = self._estimate_action_difficulty(action)
        grid_complexity = min(1.0, len(np.unique(grid)) / 10)
        return (action_diff + grid_complexity) / 2
    
    def curriculum_sampling(self, 
                            skill_level: Optional[float] = None,
                            n_examples: int = 1) -> List[TrainingExample]:
        """
        Gera exemplos com dificuldade adaptativa ao nivel do agente.
        
        Facil no inicio -> dificil conforme domina.
        """
        if skill_level is None:
            skill_level = self.skill_estimator()
        
        examples = []
        
        for _ in range(n_examples):
            # Gera puzzle com dificuldade proporcional
            if skill_level < 0.3:
                # Fase inicial: transformacoes simples, grids pequenos
                example = self._generate_simple_puzzle(grid_size=10, n_transformations=1)
            elif skill_level < 0.6:
                # Fase intermediaria: composicoes de 2-3 transformacoes
                example = self._generate_compositional_puzzle(grid_size=15, n_transformations=3)
            else:
                # Fase avancada: puzzles com padroes abstratos, alta complexidade
                example = self._generate_abstract_puzzle(grid_size=25, n_transformations=5)
            
            examples.append(example)
        
        return examples
    
    def _generate_simple_puzzle(self, 
                               grid_size: int = 10, 
                               n_transformations: int = 1) -> TrainingExample:
        """Gera puzzle simples para fase inicial"""
        grid = self._generate_base_grid(skill_level=0.2)
        
        # Uma transformacao simples
        action = random.choice(['color_shift', 'flip_h', 'flip_v', 'identity'])
        transformed = self._apply_transformation(grid, action)
        
        return TrainingExample(
            input_grid=transformed,
            target_grid=grid,
            action=action,
            reward=0.5,
            metadata={'type': 'simple', 'n_transformations': 1},
            difficulty=0.2
        )
    
    def _generate_compositional_puzzle(self,
                                      grid_size: int = 15,
                                      n_transformations: int = 3) -> TrainingExample:
        """Gera puzzle composicional para fase intermediaria"""
        grid = self._generate_base_grid(skill_level=0.5)
        
        # Compoe multiplas transformacoes
        actions = []
        current = grid.copy()
        
        for _ in range(n_transformations):
            action = random.choice(list(self.TRANSFORMATIONS.keys()))
            current = self._apply_transformation(current, action)
            actions.append(action)
        
        # Acao composta representada como string
        composite_action = '+'.join(actions)
        
        return TrainingExample(
            input_grid=current,
            target_grid=grid,
            action=composite_action,
            reward=0.7,
            metadata={'type': 'compositional', 'n_transformations': n_transformations},
            difficulty=0.5
        )
    
    def _generate_abstract_puzzle(self,
                                 grid_size: int = 25,
                                 n_transformations: int = 5,
                                 require_causal_reasoning: bool = True) -> TrainingExample:
        """Gera puzzle abstrato para fase avancada"""
        grid = self._generate_base_grid(skill_level=0.8)
        
        # Sequencia complexa de transformacoes
        actions = []
        current = grid.copy()
        
        for _ in range(n_transformations):
            action = random.choice(['rotate_90', 'rotate_180', 'rotate_270', 
                                   'mirror_diag', 'swap_quadrants'])
            current = self._apply_transformation(current, action)
            actions.append(action)
        
        composite_action = '+'.join(actions)
        
        return TrainingExample(
            input_grid=current,
            target_grid=grid,
            action=composite_action,
            reward=0.9,
            metadata={
                'type': 'abstract',
                'n_transformations': n_transformations,
                'causal_reasoning': require_causal_reasoning
            },
            difficulty=0.8
        )
    
    def update_success_rate(self, action: str, success: bool):
        """Atualiza taxa de sucesso para uma transformacao"""
        if action in self.success_rates:
            self.success_rates[action].append(1.0 if success else 0.0)
            # Mantem apenas ultimos 50
            if len(self.success_rates[action]) > 50:
                self.success_rates[action] = self.success_rates[action][-50:]
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Retorna estatisticas de geracao"""
        stats = self.generation_stats.copy()
        
        # Adiciona taxas de sucesso medias
        stats['avg_success_rates'] = {
            action: np.mean(rates) if rates else 0.5
            for action, rates in self.success_rates.items()
        }
        
        # Dificuldade media dos exemplos gerados
        if self.generated_examples:
            difficulties = [ex.difficulty for ex in self.generated_examples]
            stats['avg_difficulty'] = np.mean(difficulties)
            stats['difficulty_distribution'] = {
                'easy': sum(1 for d in difficulties if d < 0.33),
                'medium': sum(1 for d in difficulties if 0.33 <= d < 0.66),
                'hard': sum(1 for d in difficulties if d >= 0.66)
            }
        
        return stats
    
    def _update_stats(self, example: TrainingExample):
        """Atualiza estatisticas apos gerar exemplo"""
        self.generation_stats['total_generated'] += 1
        
        # Por dificuldade
        if example.difficulty < 0.33:
            self.generation_stats['by_difficulty']['easy'] += 1
        elif example.difficulty < 0.66:
            self.generation_stats['by_difficulty']['medium'] += 1
        else:
            self.generation_stats['by_difficulty']['hard'] += 1
        
        # Por transformacao
        action = example.action.split('+')[0]  # Primeira acao se composta
        if action in self.generation_stats['by_transformation']:
            self.generation_stats['by_transformation'][action] += 1
    
    def save_checkpoint(self, path: str):
        """Salva estado do gerador"""
        import json
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'generation_stats': self.generation_stats,
            'success_rates': {k: v[-50:] for k, v in self.success_rates.items()},
            'min_difficulty': self.min_difficulty,
            'max_difficulty': self.max_difficulty
        }
        
        with open(path / 'self_play_engine.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, path: str, **kwargs) -> 'SelfPlayDataGenerator':
        """Carrega estado do gerador"""
        import json
        from pathlib import Path
        
        path = Path(path)
        
        with open(path / 'self_play_engine.json', 'r') as f:
            checkpoint = json.load(f)
        
        generator = cls(**kwargs)
        generator.generation_stats = checkpoint['generation_stats']
        generator.success_rates = {k: v for k, v in checkpoint['success_rates'].items()}
        generator.min_difficulty = checkpoint['min_difficulty']
        generator.max_difficulty = checkpoint['max_difficulty']
        
        return generator


def _swap_quadrants(grid: np.ndarray) -> np.ndarray:
    """Helper: Troca quadrantes do grid"""
    h, w = grid.shape
    mid_h, mid_w = h // 2, w // 2
    
    result = grid.copy()
    
    # Quadrantes: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    q0 = grid[:mid_h, :mid_w].copy()
    q1 = grid[:mid_h, mid_w:].copy()
    q2 = grid[mid_h:, :mid_w].copy()
    q3 = grid[mid_h:, mid_w:].copy()
    
    # Troca: 0->3, 3->0, 1->2, 2->1
    result[:mid_h, :mid_w] = q3
    result[:mid_h, mid_w:] = q2
    result[mid_h:, :mid_w] = q1
    result[mid_h:, mid_w:] = q0
    
    return result
