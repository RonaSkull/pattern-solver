"""
test_curiosity_engine.py

Testes para IntrinsicCuriosityModule - Boom Catalyst #1
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Adiciona parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_genetic_baby_v4.curiosity_engine import (
    IntrinsicCuriosityModule,
    CuriosityType,
    CuriositySignal,
    ExplorationHistory
)


class MockWorldModel:
    """Mock de world model para testes"""
    
    def predict(self, state: np.ndarray, action: str) -> tuple:
        # Predicao imperfeita (simula incerteza)
        uncertainty = np.random.uniform(0.2, 0.8)
        predicted = state + np.random.normal(0, 0.5, state.shape)
        return np.clip(predicted, 0, 15).astype(int), uncertainty
    
    def compute_prediction_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        return np.mean((predicted - actual) ** 2) / (15 ** 2)


class TestCuriosityModule:
    """Testes do modulo de curiosidade"""
    
    @pytest.fixture
    def curiosity_module(self):
        return IntrinsicCuriosityModule(
            world_model=MockWorldModel(),
            curiosity_weight=0.4,
            exploration_epsilon=0.1
        )
    
    @pytest.fixture
    def sample_grid(self):
        return np.random.randint(0, 10, (10, 10))
    
    def test_compute_curiosity_signal(self, curiosity_module, sample_grid):
        """Testa computacao basica de sinal de curiosidade"""
        signal = curiosity_module.compute_curiosity_signal(
            state=sample_grid,
            action='rotate',
            predicted_state=sample_grid,
            actual_state=sample_grid
        )
        
        assert isinstance(signal, CuriositySignal)
        assert 0 <= signal.curiosity_value <= 1
        assert 0 <= signal.novelty <= 1
        assert 0 <= signal.learnability <= 1
        assert isinstance(signal.curiosity_type, CuriosityType)
    
    def test_novelty_decreases_with_repetition(self, curiosity_module, sample_grid):
        """Novelty deve diminuir com estados repetidos"""
        signal1 = curiosity_module.compute_curiosity_signal(sample_grid, 'rotate')
        signal2 = curiosity_module.compute_curiosity_signal(sample_grid, 'rotate')
        signal3 = curiosity_module.compute_curiosity_signal(sample_grid, 'rotate')
        
        # Novelty deve decrescer
        assert signal1.novelty >= signal2.novelty >= signal3.novelty
    
    def test_action_diversity_bonus(self, curiosity_module, sample_grid):
        """Acoes menos usadas devem receber bonus de diversidade"""
        # Usa mesma acao multiplas vezes
        for _ in range(10):
            curiosity_module.compute_curiosity_signal(sample_grid, 'rotate')
        
        # Nova acao deve ter maior diversity bonus
        signal_rotate = curiosity_module.compute_curiosity_signal(sample_grid, 'rotate')
        signal_flip = curiosity_module.compute_curiosity_signal(sample_grid, 'flip_h')
        
        # flip_h foi menos usada
        assert signal_flip.exploration_bonus >= signal_rotate.exploration_bonus
    
    def test_select_action_with_curiosity(self, curiosity_module, sample_grid):
        """Selecao de acao deve considerar curiosidade"""
        actions = ['rotate', 'flip_h', 'flip_v', 'identity']
        
        action, info = curiosity_module.select_action_with_curiosity(
            state=sample_grid,
            available_actions=actions
        )
        
        assert action in actions
        assert 'selected_action' in info
        assert 'curiosity_weight' in info
        assert len(info['all_actions']) == len(actions)
    
    def test_learning_progress_affects_curiosity(self, curiosity_module, sample_grid):
        """Progresso no aprendizado deve afetar curiosidade"""
        # Simula aprendizado (erro diminuindo)
        for i in range(10):
            error = 1.0 - (i * 0.1)  # Erro decrescente
            curiosity_module.history.add_prediction_error(error)
        
        signal = curiosity_module.compute_curiosity_signal(sample_grid, 'rotate')
        
        # Learning progress deve ser positivo
        assert signal.learning_progress > 0
    
    def test_statistics(self, curiosity_module, sample_grid):
        """Estatisticas devem ser computadas corretamente"""
        # Executa algumas exploracoes
        for _ in range(20):
            curiosity_module.compute_curiosity_signal(sample_grid, 'rotate')
        
        stats = curiosity_module.get_statistics()
        
        assert stats['total_explorations'] == 20
        assert 'curiosity_driven_ratio' in stats
        assert 'action_distribution' in stats
        assert 'rotate' in stats['action_distribution']
    
    def test_reset(self, curiosity_module, sample_grid):
        """Reset deve limpar historico (opcionalmente)"""
        # Adiciona dados ao historico
        for _ in range(10):
            curiosity_module.compute_curiosity_signal(sample_grid, 'rotate')
        
        # Reset com keep_history=False
        curiosity_module.reset(keep_history=False)
        
        stats = curiosity_module.get_statistics()
        assert stats['total_explorations'] == 0
    
    def test_exploration_epsilon(self, curiosity_module, sample_grid):
        """Exploration epsilon deve causar acoes aleatorias"""
        curiosity_module.exploration_epsilon = 0.5  # 50% aleatorio
        
        random_count = 0
        n_trials = 100
        
        for _ in range(n_trials):
            action, info = curiosity_module.select_action_with_curiosity(
                state=sample_grid,
                available_actions=['rotate', 'flip_h']
            )
            
            if info['selection_method'] == 'random_exploration':
                random_count += 1
        
        # Deve ter ~50% de exploracao aleatoria (com margem)
        assert 30 <= random_count <= 70


class TestExplorationHistory:
    """Testes do historico de exploracao"""
    
    def test_add_state(self):
        """Adicionar estado ao historico"""
        history = ExplorationHistory()
        history.add_state("hash1")
        history.add_state("hash2")
        
        assert len(history.state_hashes) == 2
    
    def test_compute_novelty_new_state(self):
        """Estado novo deve ter alta novidade"""
        history = ExplorationHistory()
        history.add_state("hash1")
        
        novelty = history.compute_novelty("hash2")  # Estado novo
        assert novelty == 1.0  # Maximo
    
    def test_compute_novelty_repeated_state(self):
        """Estado repetido deve ter baixa novidade"""
        history = ExplorationHistory()
        for _ in range(10):
            history.add_state("hash1")
        
        novelty = history.compute_novelty("hash1")
        assert novelty < 1.0  # Menos que maximo


class TestCuriosityIntegration:
    """Testes de integracao com agente V6"""
    
    def test_curiosity_module_exists(self):
        """Modulo de curiosidade deve existir no agente"""
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        assert hasattr(agent, 'curiosity_module')
        assert agent.curiosity_module is not None
    
    def test_developmental_curriculum_exists(self):
        """Curriculum deve existir no agente"""
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        assert hasattr(agent, 'developmental_curriculum')
        assert agent.developmental_curriculum is not None
    
    def test_self_play_engine_exists(self):
        """Self-play engine deve existir no agente"""
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        assert hasattr(agent, 'self_play_engine')
        assert agent.self_play_engine is not None
    
    def test_skill_level_estimation(self):
        """Estimativa de skill level deve funcionar"""
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        skill = agent._estimate_skill_level()
        assert 0 <= skill <= 1
    
    def test_self_play_data_generation(self):
        """Geracao de dados via self-play deve funcionar"""
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        n_examples = agent.generate_self_play_data(n_episodes=2)
        assert n_examples > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
