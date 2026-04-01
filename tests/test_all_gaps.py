"""
Comprehensive Test Suite for ARC-AGI-3 V5/V6 - All 11 Gaps
Tests each module individually and integrated
"""

import pytest
import numpy as np
import torch
from typing import Dict, List


# ============================================================================
# GAP 1: CAUSAL DISCOVERY TESTS
# ============================================================================

class TestCausalDiscovery:
    """Test Causal Discovery Engine (Gap 1)"""
    
    def test_causal_variable_creation(self):
        from arc_genetic_baby_v4.causal_discovery import CausalVariable
        var = CausalVariable("test_var", domain=np.array([0, 1, 2]))
        assert var.name == "test_var"
        assert len(var.domain) == 3
        assert var.is_intervenable == True
    
    def test_causal_graph_basic(self):
        from arc_genetic_baby_v4.causal_discovery import CausalVariable, CausalGraph
        vars = [
            CausalVariable("A", domain=np.array([0, 1])),
            CausalVariable("B", domain=np.array([0, 1])),
        ]
        graph = CausalGraph(vars)
        graph.add_edge("A", "B", mechanism='direct', strength=0.8)
        
        assert graph.graph.has_edge("A", "B")
        assert ("A", "B") in graph.hypotheses
        assert graph.hypotheses[("A", "B")].strength == 0.8
    
    def test_do_intervention(self):
        from arc_genetic_baby_v4.causal_discovery import CausalVariable, CausalGraph
        vars = [
            CausalVariable("action", domain=np.array([0, 1]), is_intervenable=True),
            CausalVariable("outcome", domain=np.array([0, 1])),
        ]
        graph = CausalGraph(vars)
        graph.add_edge("action", "outcome", strength=0.9)
        
        effects = graph.do_intervention("action", 1)
        assert "action" in effects
        assert effects["action"] == 1
    
    def test_causal_engine_initialization(self):
        from arc_genetic_baby_v4.causal_discovery import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine(grid_size=10, num_colors=4)
        assert engine.grid_size == 10
        assert engine.num_colors == 4
        assert len(engine.variables) > 0
    
    def test_arc_feature_extraction(self):
        from arc_genetic_baby_v4.causal_discovery import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine(grid_size=10, num_colors=4)
        
        # Grid simples com objeto colorido
        grid = np.zeros((10, 10), dtype=int)
        grid[3:6, 3:6] = 2  # Objeto 3x3 de cor 2
        
        obs = engine.observe(grid)
        assert obs['object_color'] == 2
        assert obs['object_size'] == 9
        assert 0 <= obs['object_position_x'] < 10


# ============================================================================
# GAP 2: SYMBOLIC ABSTRACTION TESTS
# ============================================================================

class TestSymbolicAbstraction:
    """Test Symbolic Abstraction Module (Gap 2)"""
    
    def test_symbolic_rule_creation(self):
        from arc_genetic_baby_v4.symbolic_abstraction import SymbolicRule
        rule = SymbolicRule(
            name="test_rule",
            condition=lambda x: x > 5,
            action=lambda x: x * 2,
            confidence=0.8
        )
        assert rule.confidence == 0.8
        assert rule.condition(6) == True
        assert rule.action(3) == 6
    
    def test_symbolic_module_initialization(self):
        from arc_genetic_baby_v4.symbolic_abstraction import SymbolicAbstractionModule
        module = SymbolicAbstractionModule()
        assert module is not None
    
    def test_symbol_extraction(self):
        from arc_genetic_baby_v4.symbolic_abstraction import SymbolicAbstractionModule
        module = SymbolicAbstractionModule()
        
        grid = np.zeros((10, 10), dtype=int)
        grid[2:5, 2:5] = 1  # Objeto quadrado
        
        symbols = module.extract_symbols(grid)
        assert 'objects' in symbols
        assert len(symbols['objects']) >= 1
    
    def test_rule_induction(self):
        from arc_genetic_baby_v4.symbolic_abstraction import SymbolicAbstractionModule
        module = SymbolicAbstractionModule()
        
        # Cria exemplos sintéticos
        examples = []
        for _ in range(5):
            inp = np.zeros((8, 8), dtype=int)
            inp[2:4, 2:4] = 1
            out = inp.copy()
            out[2:4, 2:4] = 2  # Troca cor
            examples.append((inp, out))
        
        rules = module.induce_rules_from_examples(examples, max_rules=3)
        assert isinstance(rules, list)


# ============================================================================
# GAP 3: COUNTERFACTUAL WORLD MODEL TESTS
# ============================================================================

class TestCounterfactual:
    """Test Counterfactual World Model (Gap 3)"""
    
    def test_counterfactual_engine_initialization(self):
        from arc_genetic_baby_v4.counterfactual import CounterfactualEngine
        engine = CounterfactualEngine(grid_size=10, num_colors=4)
        assert engine.grid_size == 10
        assert engine.num_colors == 4
    
    def test_action_simulation(self):
        from arc_genetic_baby_v4.counterfactual import CounterfactualEngine
        engine = CounterfactualEngine(grid_size=10, num_colors=4)
        
        grid = np.random.randint(0, 4, (10, 10))
        outcomes = engine.simulate_action(grid, 'rotate_90', num_samples=2)
        
        assert isinstance(outcomes, list)
        assert len(outcomes) > 0
        assert all(isinstance(o, np.ndarray) for o in outcomes)
    
    def test_world_model_forward(self):
        from arc_genetic_baby_v4.counterfactual import CounterfactualWorldModel
        import torch
        
        model = CounterfactualWorldModel(grid_size=10, num_colors=4, hidden_dim=64)
        
        state = torch.randn(1, 1, 10, 10)
        intervention = torch.randn(1, 128)
        
        output = model(state, intervention)
        assert output.shape == (1, 4, 10, 10)


# ============================================================================
# GAP 4: HIERARCHICAL PLANNER TESTS
# ============================================================================

class TestPlanner:
    """Test Hierarchical Planner (Gap 4)"""
    
    def test_planner_initialization(self):
        from arc_genetic_baby_v4.planner import HierarchicalPlanner
        planner = HierarchicalPlanner(max_depth=4)
        assert planner.max_depth == 4
    
    def test_plan_creation(self):
        from arc_genetic_baby_v4.planner import HierarchicalPlanner
        planner = HierarchicalPlanner(max_depth=3)
        
        grid = np.random.randint(0, 4, (8, 8))
        plan = planner.create_plan("test_task", grid)
        
        assert plan is not None
        assert hasattr(plan, 'steps')
    
    def test_mcts_planner(self):
        from arc_genetic_baby_v4.planner import MonteCarloTreeSearchPlanner
        planner = MonteCarloTreeSearchPlanner()
        
        assert planner is not None


# ============================================================================
# GAP 5: LEARNED ATTENTION TESTS
# ============================================================================

class TestAttention:
    """Test Learned Attention Mechanism (Gap 5)"""
    
    def test_attention_initialization(self):
        from arc_genetic_baby_v4.attention import LearnedAttentionMechanism
        attention = LearnedAttentionMechanism(grid_size=10)
        assert attention.grid_size == 10
    
    def test_attention_computation(self):
        from arc_genetic_baby_v4.attention import LearnedAttentionMechanism
        attention = LearnedAttentionMechanism(grid_size=10)
        
        grid = np.random.randint(0, 4, (10, 10))
        result = attention.compute_attention(grid)
        
        assert hasattr(result, 'spatial_attention')
        assert result.spatial_attention.shape == (10, 10)
        assert result.spatial_attention.min() >= 0
        assert result.spatial_attention.max() <= 1
    
    def test_saliency_detector(self):
        from arc_genetic_baby_v4.attention import SaliencyDetector
        detector = SaliencyDetector(grid_size=10)
        
        grid = np.zeros((10, 10), dtype=int)
        grid[5, 5] = 5  # Pixel saliente
        
        saliency = detector.compute_saliency(grid)
        assert saliency.shape == (10, 10)


# ============================================================================
# GAP 6: META-LEARNING TESTS
# ============================================================================

class TestMetaLearning:
    """Test Zero-Shot Meta-Learning (Gap 6)"""
    
    def test_meta_learner_initialization(self):
        from arc_genetic_baby_v4.meta_learning import MetaLearner
        import torch
        
        dummy_model = torch.nn.Linear(10, 5)
        learner = MetaLearner(dummy_model)
        assert learner is not None
    
    def test_task_strategy(self):
        from arc_genetic_baby_v4.meta_learning import MetaLearner
        import torch
        
        dummy_model = torch.nn.Linear(10, 5)
        learner = MetaLearner(dummy_model)
        
        strategy = learner.get_task_strategy("test_task")
        assert 'strategy' in strategy
        assert 'confidence' in strategy


# ============================================================================
# GAP 7: DEEP CAUSAL TESTS
# ============================================================================

class TestDeepCausal:
    """Test Deep Causal Reasoning (Gap 7)"""
    
    def test_deep_causal_initialization(self):
        from arc_genetic_baby_v4.deep_causal import DeepCausalEngine
        engine = DeepCausalEngine()
        assert engine is not None
    
    def test_latent_variable_inference(self):
        from arc_genetic_baby_v4.deep_causal import DeepCausalEngine
        engine = DeepCausalEngine()
        
        # Observa múltiplas vezes
        for _ in range(10):
            grid = np.random.randint(0, 4, (8, 8))
            engine.observe(grid)
        
        # Aprende estrutura
        structure = engine.learn_structure()
        assert structure is not None
    
    def test_deep_intervention(self):
        from arc_genetic_baby_v4.deep_causal import DeepCausalEngine
        engine = DeepCausalEngine()
        
        # Popula com observações
        for _ in range(15):
            grid = np.random.randint(0, 4, (8, 8))
            engine.observe(grid)
        
        engine.learn_structure()
        
        # Intervenção profunda
        intervention = {'object_position_x': 4}
        effects = engine.deep_intervention(intervention, max_order=2)
        
        assert 'observable_effects' in effects


# ============================================================================
# GAP 8: HIGH-ORDER SYMBOLIC TESTS
# ============================================================================

class TestHighOrderSymbolic:
    """Test High-Order Symbolic Abstraction (Gap 8)"""
    
    def test_concept_creator(self):
        from arc_genetic_baby_v4.high_order_symbolic import ConceptCreator, ConceptType
        creator = ConceptCreator()
        
        examples = [{'color': 1, 'size': 5}, {'color': 1, 'size': 6}]
        concept = creator.create_concept_from_examples(examples, ConceptType.OBJECT)
        
        assert concept is not None
        assert concept.concept_type == ConceptType.OBJECT
    
    def test_high_order_module(self):
        from arc_genetic_baby_v4.high_order_symbolic import HighOrderAbstractionModule
        module = HighOrderAbstractionModule()
        
        examples = []
        for _ in range(5):
            inp = np.random.randint(0, 4, (6, 6))
            out = np.rot90(inp)
            examples.append((inp, out))
        
        concepts = module.abstract_from_examples(examples)
        assert isinstance(concepts, list)


# ============================================================================
# GAP 9: METACOGNITION TESTS
# ============================================================================

class TestMetacognition:
    """Test Metacognition Module (Gap 9)"""
    
    def test_metacognition_module(self):
        from arc_genetic_baby_v4.metacognition import MetacognitionModule
        module = MetacognitionModule()
        assert module is not None
    
    def test_belief_revision(self):
        from arc_genetic_baby_v4.metacognition import MetacognitionModule, RevisionSeverity
        module = MetacognitionModule()
        
        module.initialize({'test_belief': True})
        
        # Simula falhas
        for i in range(5):
            result = module.metacognitive_step(
                success=(i % 2 == 0),
                beliefs_used=['test_belief'],
                state={},
                action='test',
                outcome={'success': (i % 2 == 0)}
            )
        
        status = module.get_epistemic_status()
        assert 'current_paradigm' in status
    
    def test_crisis_detection(self):
        from arc_genetic_baby_v4.metacognition import MetacognitiveMonitor
        monitor = MetacognitiveMonitor()
        
        # Registra múltiplas falhas
        for _ in range(5):
            monitor.record_attempt(
                success=False,
                beliefs_used=['belief1'],
                state={},
                action='test',
                outcome={'success': False}
            )
        
        is_crisis, crisis_type = monitor.detect_crisis()
        assert is_crisis == True


# ============================================================================
# GAP 10: PRODUCTIVE COMPOSITION TESTS
# ============================================================================

class TestProductiveComposition:
    """Test Productive Compositionality (Gap 10)"""
    
    def test_composition_engine(self):
        from arc_genetic_baby_v4.productive_composition import ProductiveCompositionEngine
        engine = ProductiveCompositionEngine(max_depth=5)
        assert engine is not None
    
    def test_primitive_composition(self):
        from arc_genetic_baby_v4.productive_composition import ProductiveCompositionEngine
        engine = ProductiveCompositionEngine()
        
        primitive_names = ['flip_h', 'flip_v']
        composition = engine.compose(primitive_names)
        
        assert composition is not None
    
    def test_search_composition(self):
        from arc_genetic_baby_v4.productive_composition import ProductiveCompositionEngine
        engine = ProductiveCompositionEngine()
        
        examples = [
            (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]))
            for _ in range(3)
        ]
        
        result = engine.search_composition_space(examples, max_depth=3, timeout=1.0)
        assert result is not None


# ============================================================================
# GAP 11: NATURAL INSTRUCTION TESTS
# ============================================================================

class TestNaturalInstruction:
    """Test Natural Instruction Learning (Gap 11)"""
    
    def test_instruction_module(self):
        from arc_genetic_baby_v4.natural_instruction import NaturalInstructionModule
        module = NaturalInstructionModule()
        assert module is not None
    
    def test_instruction_parsing(self):
        from arc_genetic_baby_v4.natural_instruction import NaturalInstructionModule
        module = NaturalInstructionModule()
        
        concepts = module.parse_instruction("mova o objeto para baixo")
        assert isinstance(concepts, list)
    
    def test_semantic_grounding(self):
        from arc_genetic_baby_v4.natural_instruction import NaturalInstructionModule, SemanticConcept
        module = NaturalInstructionModule()
        
        grid = np.zeros((8, 8), dtype=int)
        grid[4:6, 4:6] = 1
        
        is_present, confidence = module.ground_concept_to_grid(
            SemanticConcept.SYMMETRY, grid
        )
        assert isinstance(is_present, bool)
        assert 0 <= confidence <= 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestV5Integration:
    """Test V5 Agent Integration (Gaps 1-6)"""
    
    def test_v5_initialization(self):
        from arc_genetic_baby_v4.agent_v5 import ARCGeneticBabyV5
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV5(config)
        
        assert agent.causal_engine is not None
        assert agent.symbolic_module is not None
        assert agent.counterfactual_engine is not None
        assert agent.hierarchical_planner is not None
        assert agent.attention_mechanism is not None
        assert agent.meta_learner is not None
    
    def test_v5_step(self):
        from arc_genetic_baby_v4.agent_v5 import ARCGeneticBabyV5
        from arc_genetic_baby_v4.config import AgentConfig
        import numpy as np
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV5(config)
        
        grid = np.random.randint(0, 8, (10, 10))
        actions = ['rotate', 'flip_h']
        
        result = agent.step(grid, actions)
        assert result is not None
        assert hasattr(result, 'action')
        assert hasattr(result, 'confidence')


class TestV6Integration:
    """Test V6 Agent Integration (All 11 Gaps)"""
    
    def test_v6_initialization(self):
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        # V5 gaps
        assert agent.causal_engine is not None
        assert agent.symbolic_module is not None
        assert agent.counterfactual_engine is not None
        assert agent.hierarchical_planner is not None
        assert agent.attention_mechanism is not None
        assert agent.meta_learner is not None
        
        # V6 gaps
        assert agent.deep_causal is not None
        assert agent.high_order_symbolic is not None
        assert agent.metacognition is not None
        assert agent.productive_composition is not None
        assert agent.natural_instruction is not None
    
    def test_v6_step(self):
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        import numpy as np
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        grid = np.random.randint(0, 8, (10, 10))
        actions = ['rotate', 'flip_h', 'identity']
        
        result = agent.step(grid, actions)
        assert result is not None
        assert hasattr(result, 'action')
        assert hasattr(result, 'confidence')
    
    def test_v6_stats(self):
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        stats = agent.get_stats()
        assert stats['version'] == '6.0 (100% Edition)'
        assert 'deep_causal' in stats
        assert 'high_order_concepts' in stats
        assert 'metacognition' in stats


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance benchmarks"""
    
    def test_v6_step_latency(self):
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        import numpy as np
        import time
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        grid = np.random.randint(0, 8, (10, 10))
        actions = ['rotate', 'flip_h']
        
        # Warmup
        agent.step(grid, actions)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            agent.step(grid, actions)
        elapsed = time.time() - start
        
        fps = 10 / elapsed
        assert fps > 10  # At least 10 FPS
    
    def test_memory_usage(self):
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        import numpy as np
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        # Execute steps
        for _ in range(20):
            grid = np.random.randint(0, 8, (10, 10))
            agent.step(grid, ['rotate'])
        
        # If we get here without OOM, memory is managed
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
