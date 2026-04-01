"""
Hierarchical Planner for ARC-AGI-3
Multi-level planning from abstract goals to concrete actions
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq


class PlanLevel(Enum):
    """Hierarchy levels in planning"""
    TASK = "task"           # High-level goal
    STRATEGY = "strategy"   # Approach selection
    SUBGOAL = "subgoal"   # Intermediate objective
    ACTION = "action"     # Concrete action


@dataclass
class PlanNode:
    """Node in hierarchical plan tree"""
    level: PlanLevel
    description: str
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    children: List['PlanNode'] = field(default_factory=list)
    parent: Optional['PlanNode'] = None
    status: str = "pending"  # pending, active, completed, failed
    
    def add_child(self, child: 'PlanNode'):
        """Add child node"""
        child.parent = self
        self.children.append(child)
    
    def is_leaf(self) -> bool:
        """Check if node is leaf (action level)"""
        return len(self.children) == 0
    
    def get_depth(self) -> int:
        """Get depth in plan tree"""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1


class HierarchicalPlanner:
    """
    Hierarchical Task Network (HTN) planner for ARC-AGI-3
    
    Plans at multiple levels:
    - Task: Understand puzzle objective
    - Strategy: Select approach (rotation, color mapping, etc.)
    - Subgoal: Break down into steps
    - Action: Execute specific moves
    """
    
    def __init__(self, max_depth: int = 4, backtrack_limit: int = 100):
        self.max_depth = max_depth
        self.backtrack_limit = backtrack_limit
        
        # Plan library: reusable plan templates
        self.plan_library: Dict[str, PlanNode] = {}
        
        # Current plan
        self.current_plan: Optional[PlanNode] = None
        self.execution_stack: List[PlanNode] = []
        
        # Learning from failures
        self.failed_plans: List[Tuple[PlanNode, str]] = []
        
        self._init_plan_library()
    
    def _init_plan_library(self):
        """Initialize library of reusable plan templates"""
        
        # Template: Transform puzzle
        transform_plan = PlanNode(
            level=PlanLevel.TASK,
            description="Transform input to output pattern"
        )
        
        # Strategy: Identify pattern
        identify = PlanNode(
            level=PlanLevel.STRATEGY,
            description="Identify transformation pattern",
            preconditions=["grid_loaded"],
            postconditions=["pattern_identified"]
        )
        transform_plan.add_child(identify)
        
        # Subgoal: Extract features
        extract = PlanNode(
            level=PlanLevel.SUBGOAL,
            description="Extract key features from grid",
            preconditions=["pattern_identified"]
        )
        identify.add_child(extract)
        
        # Actions
        for action in ["scan_objects", "detect_symmetry", "find_relations"]:
            extract.add_child(PlanNode(
                level=PlanLevel.ACTION,
                description=action
            ))
        
        self.plan_library["transform"] = transform_plan
        
        # Template: Move to target
        move_plan = PlanNode(
            level=PlanLevel.TASK,
            description="Move object to target position"
        )
        
        # Strategy: Pathfinding
        pathfind = PlanNode(
            level=PlanLevel.STRATEGY,
            description="Find path to target",
            preconditions=["object_selected"]
        )
        move_plan.add_child(pathfind)
        
        # Subgoals: Navigate
        for step in ["align_x", "align_y", "adjust_precision"]:
            sub = PlanNode(
                level=PlanLevel.SUBGOAL,
                description=step
            )
            pathfind.add_child(sub)
            
            # Actions for each subgoal
            if step == "align_x":
                sub.add_child(PlanNode(
                    level=PlanLevel.ACTION,
                    description="move_left_or_right"
                ))
            elif step == "align_y":
                sub.add_child(PlanNode(
                    level=PlanLevel.ACTION,
                    description="move_up_or_down"
                ))
        
        self.plan_library["move"] = move_plan
    
    def create_plan(self, task_description: str,
                   initial_state: np.ndarray) -> PlanNode:
        """
        Create hierarchical plan for task
        
        Args:
            task_description: High-level task
            initial_state: Current grid state
            
        Returns:
            Root of plan tree
        """
        # Select appropriate template
        template = self._select_template(task_description)
        
        # Instantiate plan from template
        root = self._instantiate_plan(template, initial_state)
        
        self.current_plan = root
        return root
    
    def get_next_action(self) -> Optional[str]:
        """
        Get next concrete action to execute
        
        Returns:
            Action string or None if plan complete/failed
        """
        if not self.current_plan:
            return None
        
        # Find next pending leaf node
        next_node = self._find_next_pending(self.current_plan)
        
        if next_node:
            next_node.status = "active"
            self.execution_stack.append(next_node)
            return next_node.description
        
        return None
    
    def update_status(self, action: str, success: bool):
        """Update plan status after action execution"""
        # Find action in execution stack
        for node in reversed(self.execution_stack):
            if node.description == action:
                node.status = "completed" if success else "failed"
                
                if not success:
                    # Trigger replanning if action failed
                    self._handle_failure(node)
                
                break
    
    def backtrack(self) -> bool:
        """
        Backtrack to previous decision point
        
        Returns:
            True if backtracking successful, False if no alternatives
        """
        # Find last active node
        for node in reversed(self.execution_stack):
            if node.status == "active":
                node.status = "failed"
                
                # Try alternative
                parent = node.parent
                if parent:
                    alternatives = [c for c in parent.children 
                                  if c != node and c.status == "pending"]
                    if alternatives:
                        return True
        
        return False
    
    def _select_template(self, task_description: str) -> PlanNode:
        """Select appropriate plan template"""
        # Simple keyword matching
        if "transform" in task_description.lower():
            return self.plan_library.get("transform", self.plan_library["move"])
        elif "move" in task_description.lower():
            return self.plan_library["move"]
        
        # Default
        return list(self.plan_library.values())[0]
    
    def _instantiate_plan(self, template: PlanNode, 
                         state: np.ndarray) -> PlanNode:
        """Create concrete plan from template"""
        # Deep copy template
        root = copy.deepcopy(template)
        
        # Adapt to state (customize based on grid content)
        self._adapt_to_state(root, state)
        
        return root
    
    def _adapt_to_state(self, node: PlanNode, state: np.ndarray):
        """Adapt plan node to current state"""
        # Extract features from state
        num_colors = len(np.unique(state))
        has_objects = np.sum(state > 0) > 10
        
        # Adapt strategy based on state
        if node.level == PlanLevel.STRATEGY:
            if num_colors > 8:
                node.description += " (color-focused)"
            elif has_objects:
                node.description += " (object-focused)"
        
        # Recurse
        for child in node.children:
            self._adapt_to_state(child, state)
    
    def _find_next_pending(self, node: PlanNode) -> Optional[PlanNode]:
        """Find next pending leaf node (DFS)"""
        if node.status == "pending":
            if node.is_leaf():
                return node
            
            # Check children
            for child in node.children:
                result = self._find_next_pending(child)
                if result:
                    return result
        
        return None
    
    def _handle_failure(self, failed_node: PlanNode):
        """Handle plan failure"""
        # Record failure
        self.failed_plans.append((failed_node, "execution_failed"))
        
        # Mark parent for replanning
        parent = failed_node.parent
        if parent:
            parent.status = "pending"
            for child in parent.children:
                child.status = "pending"
    
    def get_plan_progress(self) -> Dict[str, Any]:
        """Get current plan execution progress"""
        if not self.current_plan:
            return {}
        
        total = self._count_nodes(self.current_plan)
        completed = self._count_completed(self.current_plan)
        
        return {
            'total_nodes': total,
            'completed': completed,
            'progress_pct': completed / total if total > 0 else 0,
            'current_action': self.execution_stack[-1].description 
                            if self.execution_stack else None
        }
    
    def _count_nodes(self, node: PlanNode) -> int:
        """Count total nodes in plan"""
        return 1 + sum(self._count_nodes(c) for c in node.children)
    
    def _count_completed(self, node: PlanNode) -> int:
        """Count completed nodes"""
        return int(node.status == "completed") + \
               sum(self._count_completed(c) for c in node.children)


class MonteCarloTreeSearchPlanner:
    """
    MCTS planner for complex decision making
    """
    
    def __init__(self, exploration_weight: float = 1.414):
        self.exploration_weight = exploration_weight
        self.Q: Dict[Tuple, float] = {}  # Action values
        self.N: Dict[Tuple, int] = {}    # Visit counts
        
    def search(self, state: np.ndarray, 
              available_actions: List[str],
              num_simulations: int = 100) -> str:
        """
        Run MCTS to select best action
        
        Args:
            state: Current state
            available_actions: Valid actions
            num_simulations: Number of MCTS iterations
            
        Returns:
            Best action
        """
        root_state = self._hash_state(state)
        
        for _ in range(num_simulations):
            # Selection
            path = self._select(root_state, available_actions)
            
            # Expansion
            leaf = path[-1]
            self._expand(leaf, available_actions)
            
            # Simulation (rollout)
            reward = self._simulate(leaf)
            
            # Backpropagation
            self._backpropagate(path, reward)
        
        # Select best action
        return self._best_action(root_state, available_actions)
    
    def _hash_state(self, state: np.ndarray) -> Tuple:
        """Hash state for dictionary keys"""
        return tuple(state.flatten()[:100])  # Truncate for efficiency
    
    def _select(self, root: Tuple, actions: List[str]) -> List[Tuple]:
        """Select path using UCB1"""
        path = [root]
        current = root
        
        while (current, actions[0]) in self.N:  # Expanded
            # Select child with highest UCB1
            best_action = None
            best_score = -float('inf')
            
            for action in actions:
                key = (current, action)
                if key not in self.N:
                    best_action = action
                    break
                
                # UCB1 score
                q = self.Q.get(key, 0)
                n = self.N[key]
                parent_n = sum(self.N.get((current, a), 0) for a in actions)
                
                score = q / n + self.exploration_weight * np.sqrt(np.log(parent_n) / n)
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            if best_action is None:
                break
            
            current = (current, best_action)  # Transition
            path.append(current)
        
        return path
    
    def _expand(self, leaf: Tuple, actions: List[str]):
        """Expand leaf node"""
        for action in actions:
            key = (leaf, action)
            if key not in self.N:
                self.N[key] = 0
                self.Q[key] = 0.0
    
    def _simulate(self, state: Tuple) -> float:
        """Simulate random rollout"""
        # Simplified: random reward
        return np.random.random()
    
    def _backpropagate(self, path: List[Tuple], reward: float):
        """Backpropagate reward"""
        for node in reversed(path):
            if isinstance(node, tuple) and len(node) == 2:
                # It's an action key
                key = node
                self.N[key] = self.N.get(key, 0) + 1
                self.Q[key] = self.Q.get(key, 0.0) + reward
    
    def _best_action(self, root: Tuple, actions: List[str]) -> str:
        """Select best action by visit count"""
        best_action = None
        best_count = -1
        
        for action in actions:
            key = (root, action)
            count = self.N.get(key, 0)
            if count > best_count:
                best_count = count
                best_action = action
        
        return best_action or actions[0]
