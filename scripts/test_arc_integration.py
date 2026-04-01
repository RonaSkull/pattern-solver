"""ARC-AGI-3 Integration Test Script.

Tests agent integration with ARC environment preview.
Validates: interface compliance, action validity, performance metrics.

Usage:
    python scripts/test_arc_integration.py --env preview --episodes 10
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_genetic_baby_v4 import ARCGeneticBabyV4, AgentConfig


def create_mock_arc_env():
    """Create mock ARC environment for testing."""
    class MockARCEnv:
        def __init__(self):
            self.grid_size = 10  # Match agent config
            self.num_colors = 16
            self.episode_count = 0
            
        def reset(self):
            """Start new episode."""
            self.episode_count += 1
            # Generate random ARC-like puzzle
            grid = np.random.randint(0, self.num_colors, (self.grid_size, self.grid_size))
            return grid
            
        def step(self, action: str) -> tuple:
            """Execute action and return (next_grid, reward, done, info)."""
            # Mock response - in real env this would apply transformation
            next_grid = np.random.randint(0, self.num_colors, (self.grid_size, self.grid_size))
            reward = np.random.random()
            done = reward > 0.9  # Episode ends on success
            info = {'action_valid': True}
            return next_grid, reward, done, info
            
        def get_available_actions(self) -> List[str]:
            """Get list of valid actions."""
            return ["up", "down", "left", "right", "stay", "undo", "click_0_0"]
    
    return MockARCEnv()


def test_interface_compliance(agent: ARCGeneticBabyV4, env, episodes: int = 10) -> Dict:
    """Test agent conforms to ARC-AGI-3 interface specs."""
    results = {
        'episodes_tested': 0,
        'episodes_completed': 0,
        'avg_reward': 0.0,
        'avg_steps': 0,
        'invalid_actions': 0,
        'errors': []
    }
    
    total_reward = 0.0
    total_steps = 0
    
    for ep in range(episodes):
        try:
            # Start episode
            grid = env.reset()
            episode_reward = 0.0
            steps = 0
            done = False
            
            while not done and steps < 100:  # Max 100 steps per episode
                # Get available actions
                actions = env.get_available_actions()
                
                # Agent step
                result = agent.step(grid, actions)
                
                # Validate action
                if result.action not in actions:
                    results['invalid_actions'] += 1
                    results['errors'].append(f"Episode {ep}: Invalid action {result.action}")
                    break
                
                # Execute in environment
                next_grid, reward, done, info = env.step(result.action)
                
                # Update agent (learning)
                agent.update(result.action, reward, done)
                
                episode_reward += reward
                steps += 1
                grid = next_grid
            
            results['episodes_tested'] += 1
            total_reward += episode_reward
            total_steps += steps
            
            if done:
                results['episodes_completed'] += 1
                
        except Exception as e:
            import traceback
            error_msg = f"Episode {ep}: {str(e)}\n{traceback.format_exc()}"
            results['errors'].append(error_msg)
    
    # Calculate averages
    if results['episodes_tested'] > 0:
        results['avg_reward'] = total_reward / results['episodes_tested']
        results['avg_steps'] = total_steps / results['episodes_tested']
    
    return results


def test_performance(agent: ARCGeneticBabyV4, env, duration_seconds: int = 10) -> Dict:
    """Test performance metrics (FPS, memory)."""
    results = {
        'duration_seconds': duration_seconds,
        'total_steps': 0,
        'fps': 0.0,
        'errors': []
    }
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    grid = env.reset()
    actions = env.get_available_actions()
    
    step_count = 0
    while time.time() - start_time < duration_seconds:
        try:
            result = agent.step(grid, actions)
            grid, _, done, _ = env.step(result.action)
            
            if done:
                grid = env.reset()
            
            step_count += 1
            
        except Exception as e:
            results['errors'].append(str(e))
            break
    
    end_time = time.time()
    end_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    elapsed = end_time - start_time
    results['total_steps'] = step_count
    results['fps'] = step_count / elapsed if elapsed > 0 else 0
    results['memory_mb'] = end_mem
    results['memory_delta_mb'] = end_mem - start_mem
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test ARC-AGI-3 agent integration')
    parser.add_argument('--env', type=str, default='preview', 
                       choices=['preview', 'mock'],
                       help='Environment to test against')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to test')
    parser.add_argument('--performance-test', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--output', type=str, default='integration_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print(f"🧪 ARC-AGI-3 Integration Test")
    print(f"   Environment: {args.env}")
    print(f"   Episodes: {args.episodes}")
    print()
    
    # Initialize agent
    print("🤖 Initializing agent...")
    config = AgentConfig(
        grid_size=10,  # Match mock env
        num_colors=16,
        active_inference={'num_samples': 50}  # Reduced for speed
    )
    agent = ARCGeneticBabyV4(config)
    print("   ✓ Agent ready")
    print()
    
    # Create environment
    if args.env == 'mock' or args.env == 'preview':
        env = create_mock_arc_env()
        print("🌐 Using mock ARC environment")
    
    # Run interface tests
    print("🔍 Testing interface compliance...")
    interface_results = test_interface_compliance(agent, env, args.episodes)
    print(f"   Episodes tested: {interface_results['episodes_tested']}")
    print(f"   Episodes completed: {interface_results['episodes_completed']}")
    print(f"   Avg reward: {interface_results['avg_reward']:.3f}")
    print(f"   Avg steps: {interface_results['avg_steps']:.1f}")
    print(f"   Invalid actions: {interface_results['invalid_actions']}")
    if interface_results['errors']:
        print(f"   ⚠ Errors: {len(interface_results['errors'])}")
    print()
    
    # Run performance test
    perf_results = None
    if args.performance_test:
        print("⚡ Running performance test (10 seconds)...")
        perf_results = test_performance(agent, env, duration_seconds=10)
        print(f"   Total steps: {perf_results['total_steps']}")
        print(f"   FPS: {perf_results['fps']:.1f}")
        print(f"   Memory: {perf_results['memory_mb']:.1f} MB")
        print()
    
    # Save results
    all_results = {
        'interface': interface_results,
        'performance': perf_results,
        'config': {
            'grid_size': config.grid_size,
            'num_colors': config.num_colors
        }
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"💾 Results saved to {output_path}")
    
    # Print errors if any
    if interface_results['errors']:
        print("\n❌ ERRORS DETECTED:")
        for err in interface_results['errors'][:3]:  # Show first 3
            print(f"   {err[:200]}...")  # Truncate long errors
    
    # Summary
    print("\n📊 Summary")
    print(f"   Interface test: {'✅ PASS' if not interface_results['errors'] else '❌ FAIL'}")
    if perf_results:
        fps_ok = perf_results['fps'] >= 500  # Target 500 FPS
        print(f"   Performance: {'✅ PASS' if fps_ok else '❌ FAIL'} ({perf_results['fps']:.1f} FPS)")
    
    return 0 if not interface_results['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
