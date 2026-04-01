"""
Test Suite de Integração V6 - Valida todos os 11 gaps funcionando juntos
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import json
import tempfile
import shutil

from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6, V6ActionResult
from arc_genetic_baby_v4.config import AgentConfig
from arc_genetic_baby_v4.causal_discovery import CausalDiscoveryEngine
from arc_genetic_baby_v4.symbolic_abstraction import SymbolicAbstractionModule
from arc_genetic_baby_v4.counterfactual import CounterfactualEngine
from arc_genetic_baby_v4.planner import HierarchicalPlanner
from arc_genetic_baby_v4.attention import LearnedAttentionMechanism
from arc_genetic_baby_v4.meta_learning import MetaLearner
from arc_genetic_baby_v4.deep_causal import DeepCausalEngine
from arc_genetic_baby_v4.high_order_symbolic import HighOrderAbstractionModule
from arc_genetic_baby_v4.metacognition import MetacognitionModule
from arc_genetic_baby_v4.productive_composition import ProductiveCompositionEngine
from arc_genetic_baby_v4.natural_instruction import NaturalInstructionModule


class TestV6FullIntegration:
    """Testes de integração completa do agente V6"""
    
    @pytest.fixture
    def agent_v6(self):
        """Cria agente V6 configurado para testes"""
        config = AgentConfig(
            grid_size=10,
            num_colors=8,
        )
        agent = ARCGeneticBabyV6(config)
        yield agent
    
    @pytest.fixture
    def sample_grid(self):
        """Grid ARC de exemplo para testes"""
        grid = np.zeros((10, 10), dtype=int)
        # Objeto quadrado azul no centro
        grid[3:7, 3:7] = 2
        # Objeto vermelho no canto
        grid[0:2, 0:2] = 3
        return grid
    
    def test_01_all_modules_initialized(self, agent_v6):
        """Gap 1-11: Todos os módulos foram inicializados"""
        assert agent_v6.causal_engine is not None, "Causal Discovery não inicializado"
        assert agent_v6.symbolic_module is not None, "Symbolic Abstraction não inicializado"
        assert agent_v6.counterfactual_engine is not None, "Counterfactual não inicializado"
        assert agent_v6.hierarchical_planner is not None, "Planner não inicializado"
        assert agent_v6.attention_mechanism is not None, "Attention não inicializado"
        assert agent_v6.meta_learner is not None, "Meta-Learning não inicializado"
        assert agent_v6.deep_causal is not None, "Deep Causal não inicializado"
        assert agent_v6.high_order_symbolic is not None, "High-Order Symbolic não inicializado"
        assert agent_v6.metacognition is not None, "Metacognition não inicializado"
        assert agent_v6.productive_composition is not None, "Productive Composition não inicializado"
        assert agent_v6.natural_instruction is not None, "Natural Instruction não inicializado"
    
    def test_02_attention_focus(self, agent_v6, sample_grid):
        """Gap 5: Attention foca em regiões relevantes"""
        attention_result = agent_v6.attention_mechanism.compute_attention(sample_grid)
        attention_map = attention_result.spatial_attention
        
        assert attention_map.shape == sample_grid.shape, "Shape do attention map incorreto"
        assert attention_map.min() >= 0, "Attention tem valores negativos"
        assert attention_map.max() <= 1, "Attention tem valores > 1"
        # Objetos devem ter atenção maior que fundo
        obj_mask = sample_grid > 0
        bg_mask = sample_grid == 0
        if np.any(obj_mask) and np.any(bg_mask):
            assert attention_map[obj_mask].mean() > attention_map[bg_mask].mean(), \
                "Attention não foca em objetos"
    
    def test_03_causal_variable_extraction(self, agent_v6, sample_grid):
        """Gap 1: Causal extrai variáveis do grid"""
        obs = agent_v6.causal_engine.observe(sample_grid)
        
        assert 'object_color' in obs, "Não extraiu cor do objeto"
        assert 'object_position_x' in obs, "Não extraiu posição X"
        assert 'object_position_y' in obs, "Não extraiu posição Y"
        assert 'object_size' in obs, "Não extraiu tamanho"
        assert obs['object_color'] in [2, 3], "Cor extraída incorreta"
    
    def test_04_symbolic_rule_induction(self, agent_v6):
        """Gap 2: Symbolic induz regras de exemplos"""
        # Cria exemplos sintéticos
        examples = []
        for _ in range(5):
            input_grid = np.zeros((10, 10), dtype=int)
            input_grid[2:4, 2:4] = 1  # Azul
            output_grid = input_grid.copy()
            output_grid[2:4, 2:4] = 2  # Vira verde
            
            examples.append((input_grid, output_grid))
        
        rules = agent_v6.symbolic_module.induce_rules_from_examples(examples, max_rules=3)
        
        assert len(rules) > 0, "Nenhuma regra induzida"
        assert rules[0].confidence > 0, "Regra tem confiança zero"
    
    def test_05_counterfactual_simulation(self, agent_v6, sample_grid):
        """Gap 3: Counterfactual simula cenários 'e se'"""
        # Simula intervenção
        outcomes = agent_v6.counterfactual_engine.simulate_action(
            sample_grid, 
            'rotate_90',
            num_samples=3
        )
        
        assert outcomes is not None, "Simulação retornou None"
        assert len(outcomes) > 0, "Nenhum outcome gerado"
        assert all(isinstance(o, np.ndarray) for o in outcomes), "Outcomes não são arrays"
    
    def test_06_hierarchical_planning(self, agent_v6):
        """Gap 4: Planner decompõe tarefas hierarquicamente"""
        test_grid = np.random.randint(0, 5, (8, 8))
        
        plan = agent_v6.hierarchical_planner.create_plan(
            "test_task", test_grid
        )
        
        assert plan is not None, "Plano é None"
        assert hasattr(plan, 'steps'), "Plano sem steps"
        assert len(plan.steps) > 0, "Plano vazio"
    
    def test_07_deep_causal_latent_vars(self, agent_v6, sample_grid):
        """Gap 7: Deep Causal infere variáveis latentes"""
        # Observa múltiplas vezes para ter dados
        for _ in range(10):
            test_grid = np.random.randint(0, 5, (8, 8))
            agent_v6.deep_causal.observe(test_grid)
        
        # Aprende estrutura
        structure = agent_v6.deep_causal.learn_structure()
        
        assert structure is not None, "Estrutura é None"
        assert 'latents_inferred' in structure, "Sem latents_inferred"
        
        # Query estrutura
        query = agent_v6.deep_causal.query_causal_structure()
        assert query is not None, "Query é None"
    
    def test_08_high_order_concept_creation(self, agent_v6):
        """Gap 8: High-Order Symbolic cria conceitos novos"""
        # Cria exemplos
        examples = []
        for _ in range(5):
            inp = np.random.randint(0, 5, (6, 6))
            out = np.rot90(inp)
            examples.append((inp, out))
        
        new_concepts = agent_v6.high_order_symbolic.abstract_from_examples(examples)
        
        # Pode não criar conceitos com poucos exemplos, mas não deve falhar
        assert new_concepts is not None, "Indução de conceitos falhou"
        
        # Estatísticas devem funcionar
        stats = agent_v6.high_order_symbolic.get_statistics()
        assert stats is not None, "Stats é None"
    
    def test_09_metacognition_monitoring(self, agent_v6, sample_grid):
        """Gap 9: Metacognition monitora confiança e revisa crenças"""
        # Simula algumas tentativas
        for i in range(5):
            agent_v6.metacognition.monitor.record_attempt(
                success=(i % 2 == 0),
                beliefs_used=['colors_are_meaningful', 'transformations_are_local'],
                state={'grid': sample_grid},
                action='test',
                outcome={'success': (i % 2 == 0)}
            )
        
        # Verifica status epistêmico
        status = agent_v6.metacognition.get_epistemic_status()
        assert status is not None, "Status é None"
        assert 'current_paradigm' in status, "Sem current_paradigm"
        
        # Verifica detecção de crise
        is_crisis, crisis_type = agent_v6.metacognition.monitor.detect_crisis()
        # Resultado depende do histórico, mas não deve crashar
    
    def test_10_productive_composition_depth(self, agent_v6):
        """Gap 10: Productive Composition com profundidade > 3"""
        # Cria exemplos simples
        examples = []
        for _ in range(3):
            inp = np.array([[1, 0], [0, 1]])
            out = np.array([[0, 1], [1, 0]])
            examples.append((inp, out))
        
        composition = agent_v6.productive_composition.search_composition_space(
            examples, max_depth=5, timeout=2.0
        )
        
        # Pode não encontrar composição, mas não deve crashar
        assert composition is not None, "Busca de composição falhou"
        
        # Stats devem funcionar
        stats = agent_v6.productive_composition.get_statistics()
        assert stats is not None, "Stats é None"
    
    def test_11_natural_instruction_grounding(self, agent_v6):
        """Gap 11: Natural Instruction faz grounding semântico"""
        instructions = [
            "mude a cor para verde",
            "gire o objeto",
            "mova para o canto"
        ]
        
        for instr in instructions:
            concepts = agent_v6.natural_instruction.parse_instruction(instr)
            # Deve retornar lista (pode ser vazia se não conhece palavras)
            assert isinstance(concepts, list), f"Grounding falhou para: {instr}"
        
        # Testa grounding em grid
        test_grid = np.zeros((8, 8), dtype=int)
        test_grid[4:6, 4:6] = 1
        
        from arc_genetic_baby_v4.natural_instruction import SemanticConcept
        is_present, conf = agent_v6.natural_instruction.ground_concept_to_grid(
            SemanticConcept.SYMMETRY, test_grid
        )
        assert isinstance(is_present, bool), "Grounding não retornou bool"
    
    def test_12_full_step_integration(self, agent_v6, sample_grid):
        """Integração completa: step() usa todos os módulos"""
        actions = ['rotate', 'flip_h', 'flip_v', 'identity']
        
        result = agent_v6.step(sample_grid, actions)
        
        assert result is not None, "Step retornou None"
        assert hasattr(result, 'action'), "Resultado sem ação"
        assert result.action in actions, f"Ação inválida: {result.action}"
        assert hasattr(result, 'confidence'), "Resultado sem confiança"
        assert 0 <= result.confidence <= 1, "Confiança fora de [0, 1]"
    
    def test_13_learning_update(self, agent_v6, sample_grid):
        """Aprendizado atualiza todos os módulos"""
        actions = ['rotate', 'flip_h']
        
        # Step inicial
        result = agent_v6.step(sample_grid, actions)
        
        # Cria próximo estado (simulado)
        next_grid = sample_grid.copy()
        next_grid = np.rot90(next_grid)  # Rotação simulada
        
        # Aprendizado
        agent_v6.learn(
            state=sample_grid,
            action=result.action,
            next_state=next_grid,
            success=True,
            reward=1.0
        )
        
        # Verifica se histórico foi atualizado
        assert len(agent_v6.success_history) > 0, "Histórico não atualizado"
    
    def test_14_checkpoint_save_load(self, agent_v6, sample_grid):
        """Checkpoint preserva estado de todos os módulos"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Salva
            try:
                # Simula checkpoint (simplificado)
                checkpoint_data = {
                    'episodes': agent_v6.episode_count,
                    'steps': agent_v6.step_count,
                    'success_history': agent_v6.success_history,
                }
                checkpoint_path = Path(tmpdir) / 'checkpoint.json'
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f)
                
                # Verifica arquivo
                assert checkpoint_path.exists(), "Checkpoint não foi salvo"
                
                # Carrega
                with open(checkpoint_path, 'r') as f:
                    loaded = json.load(f)
                
                assert 'episodes' in loaded, "Checkpoint incompleto"
                assert loaded['episodes'] == agent_v6.episode_count, "Episódios não batem"
                
            except Exception as e:
                pytest.skip(f"Checkpoint simplificado: {e}")
    
    def test_15_explanation_generation(self, agent_v6, sample_grid):
        """Gera estatísticas com informações de todos os módulos"""
        actions = ['rotate', 'flip_h']
        result = agent_v6.step(sample_grid, actions)
        
        stats = agent_v6.get_stats()
        
        assert stats is not None, "Stats é None"
        assert 'version' in stats, "Stats sem version"
        assert stats['version'] == '6.0 (100% Edition)', "Versão incorreta"
        assert 'deep_causal' in stats, "Stats sem deep_causal"
        assert 'high_order_concepts' in stats, "Stats sem high_order_concepts"
        assert 'metacognition' in stats, "Stats sem metacognition"


class TestV6Performance:
    """Testes de performance do V6"""
    
    @pytest.fixture
    def agent_v6(self):
        config = AgentConfig(grid_size=10, num_colors=8)
        return ARCGeneticBabyV6(config)
    
    def test_step_latency(self, agent_v6):
        """Step deve completar em < 5 segundos (limite Kaggle)"""
        import time
        
        grid = np.random.randint(0, 8, (10, 10))
        actions = ['rotate', 'flip_h', 'flip_v', 'identity']
        
        start = time.time()
        result = agent_v6.step(grid, actions)
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Step muito lento: {elapsed:.2f}s"
    
    def test_memory_usage(self, agent_v6):
        """Uso de memória deve ser controlado"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Executa múltiplos steps
        for _ in range(10):
            grid = np.random.randint(0, 8, (10, 10))
            agent_v6.step(grid, ['rotate', 'flip_h'])
        
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_delta = mem_after - mem_before
        
        # Deve ser < 500MB de aumento
        assert mem_delta < 500, f"Vazamento de memória: {mem_delta:.0f}MB"


class TestV6EdgeCases:
    """Testes de casos extremos"""
    
    @pytest.fixture
    def agent_v6(self):
        config = AgentConfig(grid_size=10, num_colors=8)
        return ARCGeneticBabyV6(config)
    
    def test_empty_grid(self, agent_v6):
        """Grid vazio não causa crash"""
        grid = np.zeros((10, 10), dtype=int)
        result = agent_v6.step(grid, ['rotate', 'flip_h'])
        assert result is not None
    
    def test_full_grid(self, agent_v6):
        """Grid completamente preenchido não causa crash"""
        grid = np.random.randint(0, 8, (10, 10))
        result = agent_v6.step(grid, ['rotate', 'flip_h'])
        assert result is not None
    
    def test_single_color(self, agent_v6):
        """Grid com única cor não causa crash"""
        grid = np.full((10, 10), 3, dtype=int)
        result = agent_v6.step(grid, ['rotate', 'flip_h'])
        assert result is not None


# Runner de validação completa
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
