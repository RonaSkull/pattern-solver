"""Main entry point for ARC Genetic Baby V4."""

import argparse
import sys
from pathlib import Path

import numpy as np

from arc_genetic_baby_v4 import ARCGeneticBabyV4, AgentConfig


def create_demo_grid(size: int = 10) -> np.ndarray:
    """Create a demo grid with a pattern."""
    grid = np.zeros((size, size), dtype=int)
    # Add a pattern (rectangle)
    grid[size//4:size*3//4, size//4:size*3//4] = 1
    return grid


def run_demo():
    """Run a demonstration of the agent."""
    print("=" * 60)
    print("ARC-AGI-3 Genetic Baby V4 - Demo")
    print("=" * 60)
    
    # Create agent
    config = AgentConfig(
        grid_size=10,
        num_colors=10,
    )
    agent = ARCGeneticBabyV4(config)
    
    print(f"\nAgent created with {len(agent.ensemble.agents)} ensemble agents")
    print(f"Developmental stage: {agent.developmental_stage}")
    
    # Create demo grid
    grid = create_demo_grid(10)
    actions = ["up", "down", "left", "right", "stay", "undo"]
    
    print(f"\nDemo grid shape: {grid.shape}")
    print(f"Unique values: {np.unique(grid)}")
    print(f"Available actions: {actions}")
    
    # Run agent step
    print("\n" + "-" * 60)
    print("Running agent step...")
    print("-" * 60)
    
    result = agent.step(grid, actions)
    
    print(f"Selected action: {result.action}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Reasoning path: {result.reasoning_path}")
    
    if result.info:
        print(f"\nAdditional info:")
        for key, value in result.info.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
    
    # Simulate learning
    print("\n" + "-" * 60)
    print("Simulating learning...")
    print("-" * 60)
    
    for i in range(10):
        next_grid = np.random.randint(0, 10, (10, 10))
        agent.learn(
            state=grid,
            action=result.action,
            next_state=next_grid,
            success=True,
            reward=1.0
        )
        grid = next_grid
        
        if i % 3 == 0:
            result = agent.step(grid, actions)
            print(f"Step {i+1}: action={result.action}, conf={result.confidence:.3f}")
    
    # Stats
    print("\n" + "-" * 60)
    print("Agent Statistics")
    print("-" * 60)
    
    stats = agent.get_stats()
    print(f"Experience count: {stats['experience_count']}")
    print(f"Developmental stage: {stats['developmental_stage']}")
    print(f"Episodic memories: {stats['memory_stats']['episodic_count']}")
    print(f"Consolidated schemas: {stats['memory_stats']['schema_count']}")
    print(f"Ensemble mean success rate: {stats['ensemble_stats']['mean_success_rate']:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def run_tests():
    """Run the test suite."""
    import pytest
    
    print("Running test suite...")
    exit_code = pytest.main(["-v", "tests/"])
    sys.exit(exit_code)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ARC-AGI-3 Genetic Baby V4 - Neuro-Cognitive Architecture"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test suite"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.demo:
        run_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
