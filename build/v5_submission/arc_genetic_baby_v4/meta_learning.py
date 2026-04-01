"""
Zero-Shot Meta-Learning for ARC-AGI-3
Fast adaptation to novel tasks
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import copy


@dataclass
class TaskEmbedding:
    """Embedding of a task for meta-learning"""
    task_id: str
    embedding: np.ndarray
    difficulty: float
    task_family: str
    success_rate: float = 0.0


class MetaLearner:
    """
    Meta-learner that adapts quickly to new tasks
    
    Implements MAML-style (Model-Agnostic Meta-Learning) adaptation
    """
    
    def __init__(self, base_model: nn.Module, 
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 num_inner_steps: int = 5):
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        
        # Task memory
        self.task_embeddings: Dict[str, TaskEmbedding] = {}
        self.task_family_stats: Dict[str, Dict] = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'avg_steps': 0
        })
        
        # Fast adaptation cache
        self.adapted_params: Dict[str, Dict] = {}
        
    def meta_train(self, task_batch: List[Tuple]):
        """
        Meta-train on batch of tasks
        
        Each task: (support_set, query_set)
        support_set: [(input, output), ...]
        query_set: [(input, output), ...]
        """
        meta_loss = 0.0
        
        for support_set, query_set in task_batch:
            # 1. Adapt to task (inner loop)
            adapted_model = self._inner_loop_adaptation(support_set)
            
            # 2. Evaluate on query set
            task_loss = self._evaluate(adapted_model, query_set)
            meta_loss += task_loss
        
        # 3. Meta-update (outer loop)
        meta_loss = meta_loss / len(task_batch)
        
        # Update base model
        # (Simplified - in practice use optimizer)
        for param in self.base_model.parameters():
            if param.grad is not None:
                param.data -= self.outer_lr * param.grad
    
    def _inner_loop_adaptation(self, support_set: List[Tuple]) -> nn.Module:
        """
        Fast adaptation to task support set
        
        Returns:
            Adapted model
        """
        # Clone model
        adapted = copy.deepcopy(self.base_model)
        
        # Inner loop updates
        for _ in range(self.num_inner_steps):
            # Compute loss on support set
            loss = 0.0
            for inp, target in support_set:
                pred = adapted(torch.FloatTensor(inp))
                loss += nn.functional.mse_loss(pred, torch.FloatTensor(target))
            
            loss = loss / len(support_set)
            
            # Manual gradient descent
            with torch.no_grad():
                for param in adapted.parameters():
                    if param.grad is not None:
                        param.data -= self.inner_lr * param.grad
        
        return adapted
    
    def _evaluate(self, model: nn.Module, query_set: List[Tuple]) -> float:
        """Evaluate adapted model on query set"""
        total_loss = 0.0
        
        with torch.no_grad():
            for inp, target in query_set:
                pred = model(torch.FloatTensor(inp))
                loss = nn.functional.mse_loss(pred, torch.FloatTensor(target))
                total_loss += loss.item()
        
        return total_loss / len(query_set)
    
    def adapt_to_new_task(self, task_id: str,
                         examples: List[Tuple],
                         task_family: str = "unknown") -> nn.Module:
        """
        Fast adaptation to a completely new task
        
        Args:
            task_id: Unique task identifier
            examples: Few-shot examples [(input, output), ...]
            task_family: Category of task (if known)
            
        Returns:
            Adapted model for this task
        """
        # Compute task embedding
        task_emb = self._compute_task_embedding(examples)
        
        # Store
        self.task_embeddings[task_id] = TaskEmbedding(
            task_id=task_id,
            embedding=task_emb,
            difficulty=self._estimate_difficulty(examples),
            task_family=task_family
        )
        
        # Check if similar task exists
        similar_task = self._find_similar_task(task_emb, task_family)
        
        if similar_task and similar_task in self.adapted_params:
            # Warm start from similar task
            adapted = self._warm_start_adaptation(examples, similar_task)
        else:
            # Cold start adaptation
            adapted = self._inner_loop_adaptation(examples)
        
        # Cache adapted parameters
        self.adapted_params[task_id] = {
            name: param.data.clone()
            for name, param in adapted.named_parameters()
        }
        
        return adapted
    
    def _compute_task_embedding(self, examples: List[Tuple]) -> np.ndarray:
        """Compute embedding vector representing task structure"""
        features = []
        
        for inp, out in examples[:3]:  # Use first 3 examples
            # Statistical features
            features.extend([
                np.mean(inp),
                np.std(inp),
                len(np.unique(inp)),
                np.mean(out),
                np.std(out),
                len(np.unique(out)),
            ])
        
        return np.array(features)
    
    def _estimate_difficulty(self, examples: List[Tuple]) -> float:
        """Estimate task difficulty"""
        if len(examples) < 2:
            return 0.5
        
        # Complexity metrics
        complexities = []
        for inp, out in examples:
            input_complexity = len(np.unique(inp)) * np.std(inp)
            output_complexity = len(np.unique(out)) * np.std(out)
            transformation_complexity = np.mean(inp != out)
            
            complexities.append(
                input_complexity + output_complexity + transformation_complexity * 10
            )
        
        return np.mean(complexities)
    
    def _find_similar_task(self, embedding: np.ndarray,
                         task_family: str) -> Optional[str]:
        """Find most similar previously seen task"""
        if not self.task_embeddings:
            return None
        
        # Filter by task family if available
        candidates = [
            tid for tid, te in self.task_embeddings.items()
            if te.task_family == task_family or task_family == "unknown"
        ]
        
        if not candidates:
            return None
        
        # Find nearest neighbor
        best_match = None
        best_distance = float('inf')
        
        for tid in candidates:
            other_emb = self.task_embeddings[tid].embedding
            dist = np.linalg.norm(embedding - other_emb)
            
            if dist < best_distance:
                best_distance = dist
                best_match = tid
        
        # Threshold for similarity
        if best_distance < 10.0:  # Tunable threshold
            return best_match
        
        return None
    
    def _warm_start_adaptation(self, examples: List[Tuple],
                              similar_task_id: str) -> nn.Module:
        """Adapt starting from similar task parameters"""
        # Load similar task parameters
        warm_params = self.adapted_params[similar_task_id]
        
        # Create model with warm start
        adapted = copy.deepcopy(self.base_model)
        
        with torch.no_grad():
            for name, param in adapted.named_parameters():
                if name in warm_params:
                    param.data = warm_params[name].clone()
        
        # Fine-tune
        for _ in range(self.num_inner_steps // 2):  # Fewer steps for warm start
            loss = 0.0
            for inp, target in examples:
                pred = adapted(torch.FloatTensor(inp))
                loss += nn.functional.mse_loss(pred, torch.FloatTensor(target))
            
            loss = loss / len(examples)
            
            with torch.no_grad():
                for param in adapted.parameters():
                    if param.grad is not None:
                        param.data -= self.inner_lr * param.grad
        
        return adapted
    
    def get_task_strategy(self, task_id: str) -> Dict[str, Any]:
        """
        Get recommended strategy for task based on meta-learning
        
        Returns:
            Dictionary with strategy recommendations
        """
        if task_id not in self.task_embeddings:
            return {'strategy': 'explore', 'confidence': 0.3}
        
        task_emb = self.task_embeddings[task_id]
        
        # Use task family statistics
        family_stats = self.task_family_stats[task_emb.task_family]
        
        if family_stats['attempts'] == 0:
            return {'strategy': 'explore', 'confidence': 0.3}
        
        success_rate = family_stats['successes'] / family_stats['attempts']
        
        if success_rate > 0.7:
            return {
                'strategy': 'exploit_family',
                'confidence': success_rate,
                'avg_steps': family_stats['avg_steps']
            }
        elif task_emb.difficulty > 0.7:
            return {
                'strategy': 'careful_exploration',
                'confidence': 0.5,
                'expected_steps': int(family_stats['avg_steps'] * 1.5)
            }
        else:
            return {
                'strategy': 'fast_exploration',
                'confidence': 0.6
            }
    
    def update_task_performance(self, task_id: str, success: bool,
                               num_steps: int, task_family: str):
        """Update performance statistics for task"""
        if task_id in self.task_embeddings:
            self.task_embeddings[task_id].success_rate = (
                0.9 * self.task_embeddings[task_id].success_rate + 0.1 * int(success)
            )
        
        # Update family stats
        stats = self.task_family_stats[task_family]
        stats['attempts'] += 1
        if success:
            stats['successes'] += 1
        
        # Moving average of steps
        stats['avg_steps'] = (
            0.9 * stats['avg_steps'] + 0.1 * num_steps
            if stats['avg_steps'] > 0 else num_steps
        )


class FastAdaptationPolicy:
    """
    Policy that uses meta-learning for fast task adaptation
    """
    
    def __init__(self, meta_learner: MetaLearner):
        self.meta_learner = meta_learner
        self.current_task_id: Optional[str] = None
        self.adaptation_history: List[Dict] = []
        
    def start_new_task(self, task_id: str, 
                      examples: List[Tuple],
                      task_family: str = "unknown"):
        """Initialize for new task"""
        self.current_task_id = task_id
        
        # Fast adaptation
        adapted_model = self.meta_learner.adapt_to_new_task(
            task_id, examples, task_family
        )
        
        # Get strategy recommendation
        strategy = self.meta_learner.get_task_strategy(task_id)
        
        self.adaptation_history.append({
            'task_id': task_id,
            'strategy': strategy,
            'num_examples': len(examples)
        })
        
        return adapted_model, strategy
    
    def select_action_with_meta_knowledge(self, state: np.ndarray,
                                         available_actions: List[str]) -> str:
        """
        Select action using meta-learned knowledge
        """
        if not self.current_task_id:
            return available_actions[0] if available_actions else "stay"
        
        # Get task strategy
        strategy = self.meta_learner.get_task_strategy(self.current_task_id)
        
        # Use strategy to guide action selection
        if strategy['strategy'] == 'exploit_family':
            # Use known good actions for this task family
            return self._select_exploitation_action(state, available_actions)
        elif strategy['strategy'] == 'explore':
            # Explore more randomly
            return self._select_exploration_action(state, available_actions)
        else:
            # Balanced approach
            return self._select_balanced_action(state, available_actions)
    
    def _select_exploitation_action(self, state: np.ndarray,
                                    actions: List[str]) -> str:
        """Select action based on prior knowledge"""
        # Simplified: prefer center movement
        return actions[len(actions) // 2] if actions else "stay"
    
    def _select_exploration_action(self, state: np.ndarray,
                                  actions: List[str]) -> str:
        """Select action for exploration"""
        # Random with slight bias toward novel actions
        probs = np.ones(len(actions)) / len(actions)
        return np.random.choice(actions, p=probs)
    
    def _select_balanced_action(self, state: np.ndarray,
                               actions: List[str]) -> str:
        """Balance exploration and exploitation"""
        # Epsilon-greedy
        if np.random.random() < 0.3:
            return self._select_exploration_action(state, actions)
        else:
            return self._select_exploitation_action(state, actions)
    
    def finish_task(self, success: bool, num_steps: int,
                   task_family: str = "unknown"):
        """Record task completion"""
        if self.current_task_id:
            self.meta_learner.update_task_performance(
                self.current_task_id, success, num_steps, task_family
            )
