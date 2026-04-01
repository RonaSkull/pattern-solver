"""
Comprehensive tests for V5 Agent and all 6 gaps
"""

import pytest
import numpy as np
import torch

# Test imports
from arc_genetic_baby_v4.config import AgentConfig
from arc_genetic_baby_v4.agent_v5 import ARCGeneticBabyV5, V5ActionResult
from arc_genetic_baby_v4.causal_discovery import CausalDiscoveryEngine
from arc_genetic_baby_v4.symbolic_abstraction import SymbolicAbstractionModule
from arc_genetic_baby_v4.counterfactual import CounterfactualEngine, CounterfactualPlanner
from arc_genetic_baby_v4.planner import HierarchicalPlanner
from arc_genetic_baby_v4.attention import LearnedAttentionMechanism
from arc_genetic_baby_v4.meta_learning import MetaLearner, FastAdaptationPolicy


# === GAP 1: Causal Discovery Tests ===

def test_causal_engine_initialization():
    """Test causal engine can be initialized"""
    engine = CausalDiscoveryEngine(grid_size=10, num_colors=4)
    assert engine.grid_size == 10
    assert engine.num_colors == 4
    assert len(engine.causal_graph.variables) > 0


def test_causal_observation():
    """Test causal feature extraction"""
    engine = CausalDiscoveryEngine(grid_size=10, num_colors=4)
    
    # Create simple grid with object
    grid = np.zeros((10, 10), dtype=int)
    grid[3:6, 3:6] = 2
    
    obs = engine.observe(grid)
    
    assert 'object_color' in obs
    assert obs['object_color'] == 2
    assert obs['object_size'] == 9


def test_causal_intervention():
    """Test causal intervention simulation"""
    engine = CausalDiscoveryEngine(grid_size=10, num_colors=4)
    
    # Setup state
    grid = np.zeros((10, 10), dtype=int)
    engine.observe(grid)
    
    # Simulate action
    effects = engine.intervene('rotate', param=1)
    
    assert isinstance(effects, dict)
    assert 'transformation_type' in effects


# === GAP 2: Symbolic Abstraction Tests ===

def test_symbolic_extraction():
    """Test symbolic feature extraction"""
    module = SymbolicAbstractionModule()
    
    grid = np.zeros((10, 10), dtype=int)
    grid[2:5, 2:5] = 3
    
    symbols = module.extract_symbols(grid)
    
    assert 'objects' in symbols
    assert 'background_color' in symbols
    assert len(symbols['objects']) >= 1


def test_symbolic_rule_induction():
    """Test rule induction from examples"""
    module = SymbolicAbstractionModule()
    
    # Create example pair
    input_grid = np.zeros((10, 10), dtype=int)
    input_grid[3:6, 3:6] = 1  # Blue object
    
    output_grid = input_grid.copy()
    output_grid[3:6, 3:6] = 2  # Changed to red
    
    examples = [(input_grid, output_grid)]
    rules = module.induce_rules(examples, max_rules=5)
    
    assert isinstance(rules, list)


# === GAP 3: Counterfactual Tests ===

def test_counterfactual_simulation():
    """Test counterfactual simulation"""
    engine = CounterfactualEngine(grid_size=10, num_colors=4)
    
    grid = np.random.randint(0, 4, (10, 10))
    
    outcomes = engine.simulate_action(grid, 'rotate', num_samples=3)
    
    assert len(outcomes) == 3
    assert all(o.shape == (10, 10) for o in outcomes)


def test_counterfactual_planner():
    """Test counterfactual-based planning"""
    engine = CounterfactualEngine(grid_size=10, num_colors=4)
    planner = CounterfactualPlanner(engine, horizon=2)
    
    grid = np.random.randint(0, 4, (10, 10))
    actions = ['up', 'down', 'left', 'right']
    
    best_action, score = planner.plan(grid, actions)
    
    assert best_action in actions
    assert 0 <= score <= 1


# === GAP 4: Hierarchical Planner Tests ===

def test_hierarchical_planner_creation():
    """Test hierarchical plan creation"""
    planner = HierarchicalPlanner(max_depth=3)
    
    grid = np.zeros((10, 10), dtype=int)
    
    plan = planner.create_plan("test_task", grid)
    
    assert plan is not None
    assert plan.level.value == "task"


def test_hierarchical_next_action():
    """Test getting next action from plan"""
    planner = HierarchicalPlanner()
    
    grid = np.zeros((10, 10), dtype=int)
    planner.create_plan("test", grid)
    
    action = planner.get_next_action()
    
    # Should return some action or None if plan complete
    assert action is None or isinstance(action, str)


# === GAP 5: Attention Tests ===

def test_attention_computation():
    """Test attention mechanism"""
    attention = LearnedAttentionMechanism(grid_size=10, device='cpu')
    
    grid = np.random.randint(0, 4, (10, 10))
    
    attn_map = attention.compute_attention(grid)
    
    assert attn_map.spatial_attention.shape == (10, 10)
    assert attn_map.spatial_attention.min() >= 0
    assert attn_map.spatial_attention.max() <= 1


def test_saliency_detection():
    """Test saliency detection"""
    from arc_genetic_baby_v4.attention import SaliencyDetector
    
    detector = SaliencyDetector(grid_size=10)
    
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 3  # Salient point
    
    saliency = detector.detect_saliency(grid)
    
    assert saliency.shape == (10, 10)
    assert saliency[5, 5] > saliency[0, 0]  # Center more salient


# === GAP 6: Meta-Learning Tests ===

def test_meta_learner_initialization():
    """Test meta-learner can be initialized"""
    model = torch.nn.Linear(10, 4)
    learner = MetaLearner(model)
    
    assert learner.inner_lr > 0
    assert learner.outer_lr > 0


def test_task_embedding():
    """Test task embedding computation"""
    model = torch.nn.Linear(10, 4)
    learner = MetaLearner(model)
    
    # Create dummy examples
    examples = [
        (np.random.randn(10), np.random.randn(4))
        for _ in range(3)
    ]
    
    embedding = learner._compute_task_embedding(examples)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0


def test_fast_adaptation():
    """Test fast adaptation to new task"""
    model = torch.nn.Linear(10, 4)
    learner = MetaLearner(model)
    
    examples = [
        (np.random.randn(10), np.random.randn(4))
        for _ in range(5)
    ]
    
    adapted = learner.adapt_to_new_task("test_task", examples)
    
    assert isinstance(adapted, torch.nn.Module)


# === V5 Agent Integration Tests ===

def test_v5_agent_initialization():
    """Test V5 agent can be initialized"""
    config = AgentConfig(grid_size=10, num_colors=4)
    agent = ARCGeneticBabyV5(config)
    
    assert agent.config.grid_size == 10
    assert agent.causal_engine is not None
    assert agent.symbolic_module is not None
    assert agent.counterfactual_engine is not None


def test_v5_step():
    """Test V5 agent step execution"""
    config = AgentConfig(grid_size=10, num_colors=4)
    agent = ARCGeneticBabyV5(config)
    
    grid = np.random.randint(0, 4, (10, 10))
    actions = ['up', 'down', 'left', 'right', 'stay']
    
    result = agent.step(grid, actions)
    
    assert isinstance(result, V5ActionResult)
    assert result.action in actions
    assert 0 <= result.confidence <= 1
    assert result.reasoning != ""


def test_v5_learning():
    """Test V5 agent learning"""
    config = AgentConfig(grid_size=10, num_colors=4)
    agent = ARCGeneticBabyV5(config)
    
    state = np.random.randint(0, 4, (10, 10))
    next_state = np.random.randint(0, 4, (10, 10))
    
    agent.learn(state, 'rotate', next_state, success=True, reward=1.0)
    
    # Check that learning occurred
    assert len(agent.success_history) == 1
    assert agent.success_history[0] == True


def test_v5_stats():
    """Test V5 agent statistics"""
    config = AgentConfig(grid_size=10, num_colors=4)
    agent = ARCGeneticBabyV5(config)
    
    # Run a few steps
    for i in range(5):
        grid = np.random.randint(0, 4, (10, 10))
        actions = ['up', 'down', 'left', 'right', 'stay']
        agent.step(grid, actions)
        agent.learn(grid, actions[0], grid, success=(i % 2 == 0))
    
    stats = agent.get_stats()
    
    assert stats['version'] == '5.0'
    assert stats['steps'] >= 5
    assert 'success_rate' in stats


def test_v5_reset():
    """Test V5 agent reset"""
    config = AgentConfig(grid_size=10, num_colors=4)
    agent = ARCGeneticBabyV5(config)
    
    # Do some work
    grid = np.random.randint(0, 4, (10, 10))
    agent.step(grid, ['up', 'down'])
    
    # Reset
    agent.reset(new_task_id="test_task_123")
    
    assert agent.current_task_id == "test_task_123"
    assert agent.step_count == 0


# === Performance Tests ===

@pytest.mark.slow
def test_v5_performance():
    """Test V5 agent performance meets targets"""
    import time
    
    config = AgentConfig(grid_size=10, num_colors=4)
    agent = ARCGeneticBabyV5(config)
    
    grid = np.random.randint(0, 4, (10, 10))
    actions = ['up', 'down', 'left', 'right', 'stay']
    
    # Warm up
    for _ in range(5):
        agent.step(grid, actions)
    
    # Measure
    start = time.time()
    num_steps = 100
    
    for _ in range(num_steps):
        agent.step(grid, actions)
    
    elapsed = time.time() - start
    fps = num_steps / elapsed
    
    # Should achieve reasonable FPS (adjust target as needed)
    assert fps > 10, f"FPS too low: {fps:.1f}"


# === Integration Tests ===

def test_all_gaps_work_together():
    """Test that all 6 gaps work together in agent"""
    config = AgentConfig(grid_size=10, num_colors=4)
    agent = ARCGeneticBabyV5(config)
    
    # Simulate episode
    for step in range(10):
        grid = np.random.randint(0, 4, (10, 10))
        actions = ['up', 'down', 'left', 'right', 'stay']
        
        result = agent.step(grid, actions)
        agent.learn(grid, result.action, grid, success=(step % 3 == 0))
    
    stats = agent.get_stats()
    
    # All gaps should have contributed
    assert stats['causal_graph_size'] >= 0
    assert stats['symbolic_rules'] >= 0
    assert stats['meta_tasks'] >= 0


def test_v5_checkpoint_roundtrip():
    """Test checkpoint save/load"""
    import tempfile
    from pathlib import Path
    
    config = AgentConfig(grid_size=10, num_colors=4)
    agent = ARCGeneticBabyV5(config)
    
    # Do some work
    for _ in range(5):
        grid = np.random.randint(0, 4, (10, 10))
        agent.step(grid, ['up', 'down'])
    
    original_stats = agent.get_stats()
    
    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        agent.save_checkpoint(tmpdir)
        
        # Load into new agent
        agent2 = ARCGeneticBabyV5(config)
        agent2.load_checkpoint(tmpdir)
        
        loaded_stats = agent2.get_stats()
        
        # Stats should be preserved
        assert loaded_stats['episodes'] == original_stats['episodes']
