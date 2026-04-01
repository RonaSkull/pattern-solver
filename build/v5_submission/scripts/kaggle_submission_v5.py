"""
ARC-AGI-3 Kaggle Submission Script V5.0
Complete agent with all 6 critical gaps integrated
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from arc_genetic_baby_v4.config import AgentConfig
from arc_genetic_baby_v4.agent_v5 import ARCGeneticBabyV5


class ARCSubmissionAgent:
    """Kaggle-compatible submission wrapper for V5 Agent"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.agent = ARCGeneticBabyV5(self.config)
        self.submission_data: Dict[str, List[List[List[int]]]] = {}
        
    def solve_task(self, task_id: str, 
                   train_examples: List[Dict],
                   test_input: np.ndarray) -> List[List[int]]:
        """
        Solve a single ARC task
        
        Args:
            task_id: Unique task identifier
            train_examples: List of {'input': grid, 'output': grid}
            test_input: Test input grid
            
        Returns:
            Predicted output grid
        """
        # Reset agent for new task
        self.agent.reset(new_task_id=task_id)
        
        # Learn from training examples
        for example in train_examples:
            inp = np.array(example['input'])
            out = np.array(example['output'])
            
            # Observe causal state
            self.agent.causal_engine.observe(inp)
            
            # Learn pattern
            self.agent.learn(inp, 'train_example', out, success=True, reward=1.0)
        
        # Extract symbols from test input
        symbols = self.agent.symbolic_module.extract_symbols(test_input)
        
        # Try symbolic rules first
        result_grid, applied_rule = self.agent.symbolic_module.apply_rules(test_input)
        
        if applied_rule and applied_rule.confidence > 0.6:
            # Use symbolic prediction
            return result_grid.tolist()
        
        # Fall back to counterfactual planning
        available_actions = self._get_available_actions(test_input)
        
        # Get hierarchical plan
        plan = self.agent.hierarchical_planner.create_plan(
            f"solve_{task_id}", test_input
        )
        
        # Simulate actions
        best_action = None
        best_score = -1
        
        for action in available_actions:
            outcomes = self.agent.counterfactual_engine.simulate_action(
                test_input, action, num_samples=3
            )
            
            # Score outcomes
            scores = []
            for outcome in outcomes:
                score = self._score_output(outcome, train_examples)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_action = action
        
        # Generate output based on best action
        if best_action:
            final_outcomes = self.agent.counterfactual_engine.simulate_action(
                test_input, best_action, num_samples=1
            )
            if final_outcomes:
                return final_outcomes[0].tolist()
        
        # Fallback: return test input with minor modifications
        # This ensures we always return a valid grid
        return self._fallback_output(test_input, train_examples)
    
    def _get_available_actions(self, grid: np.ndarray) -> List[str]:
        """Get list of available actions for this grid"""
        actions = [
            'identity',
            'rotate_90',
            'rotate_180',
            'flip_horizontal',
            'flip_vertical',
            'transpose',
            'invert_colors',
            'fill_background',
        ]
        return actions
    
    def _score_output(self, output: np.ndarray, 
                     train_examples: List[Dict]) -> float:
        """Score output against training patterns"""
        score = 0.5  # Base score
        
        # Check similarity to training outputs
        for example in train_examples:
            train_out = np.array(example['output'])
            
            # Size similarity
            if output.shape == train_out.shape:
                score += 0.2
            
            # Color palette similarity
            out_colors = set(np.unique(output))
            train_colors = set(np.unique(train_out))
            color_overlap = len(out_colors & train_colors)
            color_score = color_overlap / max(len(out_colors), len(train_colors), 1)
            score += 0.3 * color_score
        
        # Prefer non-empty outputs
        if np.any(output > 0):
            score += 0.1
        
        return min(1.0, score)
    
    def _fallback_output(self, test_input: np.ndarray,
                        train_examples: List[Dict]) -> List[List[int]]:
        """Generate fallback output when no rule applies"""
        if not train_examples:
            return test_input.tolist()
        
        # Use first training output as template
        template = np.array(train_examples[0]['output'])
        
        # Resize test input to match template if needed
        if test_input.shape != template.shape:
            from scipy.ndimage import zoom
            zoom_y = template.shape[0] / test_input.shape[0]
            zoom_x = template.shape[1] / test_input.shape[1]
            resized = zoom(test_input, (zoom_y, zoom_x), order=0)
            
            # Ensure exact shape match
            result = np.zeros_like(template)
            min_h = min(resized.shape[0], template.shape[0])
            min_w = min(resized.shape[1], template.shape[1])
            result[:min_h, :min_w] = resized[:min_h, :min_w]
        else:
            result = test_input.copy()
        
        return result.tolist()
    
    def generate_submission(self, data_path: str, 
                           output_path: str = 'submission.json'):
        """
        Generate Kaggle submission from evaluation data
        
        Args:
            data_path: Path to ARC evaluation data directory
            output_path: Path for output submission JSON
        """
        data_path = Path(data_path)
        
        # Load evaluation tasks
        eval_tasks = self._load_eval_tasks(data_path)
        
        print(f"Processing {len(eval_tasks)} tasks...")
        
        # Solve each task
        for i, (task_id, task_data) in enumerate(eval_tasks.items()):
            print(f"  [{i+1}/{len(eval_tasks)}] {task_id}...", end='', flush=True)
            
            start_time = time.time()
            
            try:
                # Extract examples
                train_examples = task_data.get('train', [])
                test_inputs = task_data.get('test', [])
                
                # Solve each test input
                predictions = []
                for test_input_data in test_inputs:
                    test_input = np.array(test_input_data['input'])
                    
                    prediction = self.solve_task(
                        task_id, train_examples, test_input
                    )
                    predictions.append(prediction)
                
                # Store predictions
                self.submission_data[task_id] = predictions
                
                elapsed = time.time() - start_time
                print(f" ✓ ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f" ✗ Error: {e}")
                # Add empty prediction on error
                self.submission_data[task_id] = [[]]
        
        # Save submission
        with open(output_path, 'w') as f:
            json.dump(self.submission_data, f)
        
        print(f"\nSubmission saved to: {output_path}")
        print(f"Total tasks: {len(self.submission_data)}")
        
        return self.submission_data
    
    def _load_eval_tasks(self, data_path: Path) -> Dict[str, Any]:
        """Load ARC evaluation tasks from directory"""
        tasks = {}
        
        # Look for JSON files in data path
        for json_file in data_path.glob('*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Handle both single task and multiple tasks formats
                if isinstance(data, dict) and 'train' in data:
                    # Single task
                    tasks[json_file.stem] = data
                elif isinstance(data, dict):
                    # Multiple tasks
                    tasks.update(data)
        
        return tasks
    
    def validate_submission(self, submission_path: str) -> bool:
        """Validate submission format"""
        with open(submission_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            print("ERROR: Submission must be a JSON object")
            return False
        
        for task_id, predictions in data.items():
            if not isinstance(predictions, list):
                print(f"ERROR: Task {task_id} predictions must be a list")
                return False
            
            for pred in predictions:
                if not isinstance(pred, list):
                    print(f"ERROR: Task {task_id} prediction must be 2D array")
                    return False
        
        print(f"✓ Valid submission with {len(data)} tasks")
        return True


def main():
    """Main entry point for Kaggle submission"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ARC-AGI-3 Kaggle Submission Generator V5.0'
    )
    parser.add_argument(
        '--data', type=str, default='data/evaluation',
        help='Path to evaluation data'
    )
    parser.add_argument(
        '--output', type=str, default='submission.json',
        help='Output submission file'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Validate submission after generation'
    )
    
    args = parser.parse_args()
    
    # Create agent
    config = AgentConfig(
        grid_size=30,  # ARC typical size
        num_colors=10
    )
    
    submission_agent = ARCSubmissionAgent(config)
    
    # Generate submission
    submission_agent.generate_submission(args.data, args.output)
    
    # Validate if requested
    if args.validate:
        submission_agent.validate_submission(args.output)
    
    print("\n✅ ARC-AGI-3 V5 Submission Complete!")


if __name__ == '__main__':
    main()
