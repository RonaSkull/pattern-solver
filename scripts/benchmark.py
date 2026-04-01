"""Benchmark Script for ARC-AGI-3 Agent.

Measures performance metrics for Kaggle submission compliance:
- FPS (frames/actions per second)
- Memory usage
- Episode completion time

Usage:
    python scripts/benchmark.py --target-fps 1000 --memory-limit 2GB
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict

import numpy as np
import psutil
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_genetic_baby_v4 import ARCGeneticBabyV4, AgentConfig


def parse_memory_limit(limit_str: str) -> int:
    """Parse memory limit string (e.g., '2GB' -> 2048 MB)."""
    limit_str = limit_str.upper()
    if limit_str.endswith('GB'):
        return int(limit_str[:-2]) * 1024
    elif limit_str.endswith('MB'):
        return int(limit_str[:-2])
    else:
        return int(limit_str)


def benchmark_fps(agent: ARCGeneticBabyV4, duration: int = 10) -> Dict[str, Any]:
    """Benchmark FPS over fixed duration."""
    process = psutil.Process(os.getpid())
    
    # Warmup
    grid = np.random.randint(0, 16, (10, 10))  # Consistent grid size
    actions = ["up", "down", "left", "right", "stay"]
    for _ in range(10):
        agent.step(grid, actions)
    
    # Benchmark
    start_mem = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    steps = 0
    while time.time() - start_time < duration:
        result = agent.step(grid, actions)
        steps += 1
    
    end_time = time.time()
    end_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    elapsed = end_time - start_time
    fps = steps / elapsed
    
    return {
        'duration': duration,
        'total_steps': steps,
        'fps': round(fps, 2),
        'memory_start_mb': round(start_mem, 1),
        'memory_end_mb': round(end_mem, 1),
        'memory_delta_mb': round(end_mem - start_mem, 1)
    }


def benchmark_episode(agent: ARCGeneticBabyV4, num_episodes: int = 5) -> Dict[str, Any]:
    """Benchmark episode completion metrics."""
    episode_times = []
    episode_steps = []
    
    for ep in range(num_episodes):
        grid = np.random.randint(0, 16, (10, 10))  # Consistent grid size
        actions = ["up", "down", "left", "right", "stay"]
        
        start_time = time.time()
        steps = 0
        
        # Simulate episode (30 steps max)
        for _ in range(30):
            result = agent.step(grid, actions)
            steps += 1
            # Simulate state change
            grid = np.random.randint(0, 16, (64, 64))
        
        end_time = time.time()
        
        episode_times.append(end_time - start_time)
        episode_steps.append(steps)
    
    return {
        'num_episodes': num_episodes,
        'avg_episode_time_ms': round(sum(episode_times) / len(episode_times) * 1000, 1),
        'avg_episode_steps': round(sum(episode_steps) / len(episode_steps), 1),
        'total_time_s': round(sum(episode_times), 2)
    }


def generate_report(results: Dict, target_fps: int, memory_limit_mb: int) -> str:
    """Generate benchmark report."""
    fps_result = results['fps_benchmark']
    episode_result = results['episode_benchmark']
    
    fps_ok = fps_result['fps'] >= target_fps
    memory_ok = fps_result['memory_end_mb'] < memory_limit_mb
    
    report = f"""# ARC-AGI-3 Agent Benchmark Report

## Configuration
- Target FPS: {target_fps}
- Memory Limit: {memory_limit_mb} MB
- Grid Size: {results['config']['grid_size']}
- Num Colors: {results['config']['num_colors']}

## FPS Benchmark ({fps_result['duration']}s)
- **FPS: {fps_result['fps']:.1f}** {'✅ PASS' if fps_ok else '❌ FAIL'}
- Total Steps: {fps_result['total_steps']:,}
- Memory Start: {fps_result['memory_start_mb']:.1f} MB
- Memory End: {fps_result['memory_end_mb']:.1f} MB
- Memory Delta: {fps_result['memory_delta_mb']:.1f} MB {'✅ PASS' if memory_ok else '❌ FAIL'}

## Episode Benchmark ({episode_result['num_episodes']} episodes)
- Avg Episode Time: {episode_result['avg_episode_time_ms']:.1f} ms
- Avg Episode Steps: {episode_result['avg_episode_steps']:.1f}
- Total Time: {episode_result['total_time_s']:.2f} s

## Compliance
- FPS Target: {'✅ PASS' if fps_ok else '❌ FAIL'} ({fps_result['fps']:.1f} >= {target_fps})
- Memory Limit: {'✅ PASS' if memory_ok else '❌ FAIL'} ({fps_result['memory_end_mb']:.1f} < {memory_limit_mb})

## Conclusion
{'✅ Agent meets Kaggle submission requirements!' if fps_ok and memory_ok else '❌ Agent needs optimization before submission.'}
"""
    return report


def main():
    parser = argparse.ArgumentParser(description='Benchmark ARC-AGI-3 agent performance')
    parser.add_argument('--target-fps', type=int, default=1000,
                       help='Target FPS (default: 1000)')
    parser.add_argument('--memory-limit', type=str, default='2GB',
                       help='Memory limit (default: 2GB)')
    parser.add_argument('--report', type=str, default='performance_report.md',
                       help='Output report file')
    parser.add_argument('--json', type=str, default='benchmark_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    memory_limit_mb = parse_memory_limit(args.memory_limit)
    
    print(f"📊 ARC-AGI-3 Agent Benchmark")
    print(f"   Target FPS: {args.target_fps}")
    print(f"   Memory Limit: {memory_limit_mb} MB")
    print()
    
    # Initialize agent
    print("🤖 Initializing agent...")
    config = AgentConfig(grid_size=10, num_colors=16)  # Consistent with tests
    agent = ARCGeneticBabyV4(config)
    print("   ✓ Agent ready\n")
    
    # FPS Benchmark
    print("⚡ Running FPS benchmark (10 seconds)...")
    fps_results = benchmark_fps(agent, duration=10)
    print(f"   FPS: {fps_results['fps']:.1f}")
    print(f"   Memory: {fps_results['memory_end_mb']:.1f} MB")
    print()
    
    # Episode Benchmark
    print("🎮 Running episode benchmark (5 episodes)...")
    episode_results = benchmark_episode(agent, num_episodes=5)
    print(f"   Avg time: {episode_results['avg_episode_time_ms']:.1f} ms/episode")
    print()
    
    # Compile results
    all_results = {
        'config': asdict(config),
        'fps_benchmark': fps_results,
        'episode_benchmark': episode_results,
        'targets': {
            'fps': args.target_fps,
            'memory_mb': memory_limit_mb
        }
    }
    
    # Save JSON
    json_path = Path(args.json)
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"💾 JSON saved to {json_path}")
    
    # Generate and save report
    report = generate_report(all_results, args.target_fps, memory_limit_mb)
    report_path = Path(args.report)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Report saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    fps_ok = fps_results['fps'] >= args.target_fps
    mem_ok = fps_results['memory_end_mb'] < memory_limit_mb
    
    if fps_ok and mem_ok:
        print("✅ BENCHMARK PASSED")
    else:
        print("❌ BENCHMARK FAILED")
    
    print(f"   FPS: {fps_results['fps']:.1f} / {args.target_fps} {'✅' if fps_ok else '❌'}")
    print(f"   Memory: {fps_results['memory_end_mb']:.1f} / {memory_limit_mb} MB {'✅' if mem_ok else '❌'}")
    
    return 0 if fps_ok and mem_ok else 1


if __name__ == '__main__':
    sys.exit(main())
