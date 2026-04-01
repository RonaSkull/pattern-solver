"""
Productive Compositionality for ARC-AGI-3
Unlimited depth composition with intelligent pruning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import itertools
import heapq


class CompositionType(Enum):
    """Types of composition"""
    SEQUENTIAL = auto()      # A then B then C
    PARALLEL = auto()        # A and B simultaneously
    HIERARCHICAL = auto()    # A(B(C(x)))
    CONDITIONAL = auto()     # if A then B else C
    RECURSIVE = auto()       # A(A(A(x)))


@dataclass
class ComposablePrimitive:
    """A primitive that can be composed"""
    name: str
    function: Callable
    input_type: str
    output_type: str
    complexity: int = 1
    preconditions: List[str] = field(default_factory=list)
    
    def can_compose_with(self, other: 'ComposablePrimitive') -> bool:
        """Check if this can be composed with another primitive"""
        # Type compatibility
        if self.output_type != other.input_type:
            return False
        
        # Complexity limit (prevent infinite expansion)
        if self.complexity + other.complexity > 20:
            return False
        
        return True
    
    def compose_sequential(self, other: 'ComposablePrimitive') -> 'ComposablePrimitive':
        """Create A then B"""
        def composed_fn(x):
            intermediate = self.function(x)
            return other.function(intermediate)
        
        return ComposablePrimitive(
            name=f"{self.name}_then_{other.name}",
            function=composed_fn,
            input_type=self.input_type,
            output_type=other.output_type,
            complexity=self.complexity + other.complexity,
            preconditions=self.preconditions + other.preconditions
        )


@dataclass
class CompositionNode:
    """Node in composition tree"""
    primitive: ComposablePrimitive
    children: List['CompositionNode'] = field(default_factory=list)
    depth: int = 0
    score: float = 0.0
    composition_type: CompositionType = CompositionType.SEQUENTIAL
    
    def total_complexity(self) -> int:
        """Total complexity of this composition"""
        base = self.primitive.complexity
        children_complexity = sum(c.total_complexity() for c in self.children)
        return base + children_complexity
    
    def evaluate(self, input_data: Any) -> Any:
        """Evaluate composition on input"""
        if not self.children:
            return self.primitive.function(input_data)
        
        if self.composition_type == CompositionType.SEQUENTIAL:
            result = input_data
            for child in self.children:
                result = child.evaluate(result)
            return self.primitive.function(result)
        
        elif self.composition_type == CompositionType.PARALLEL:
            results = [child.evaluate(input_data) for child in self.children]
            # Combine results
            return self.primitive.function(results)
        
        elif self.composition_type == CompositionType.HIERARCHICAL:
            # Innermost first
            result = self.children[-1].evaluate(input_data) if self.children else input_data
            for child in reversed(self.children[:-1]):
                result = child.evaluate(result)
            return self.primitive.function(result)
        
        else:
            # Default
            return self.primitive.function(input_data)


class ProductiveCompositionEngine:
    """
    Engine for unlimited-depth productive composition
    
    Key insight: Humans don't search all combinations.
    They use structure, analogy, and heuristics to guide composition.
    """
    
    def __init__(self, max_depth: int = 10, beam_width: int = 5):
        self.max_depth = max_depth
        self.beam_width = beam_width
        
        self.primitives: Dict[str, ComposablePrimitive] = {}
        self.composition_cache: Dict[str, Any] = {}
        self.successful_compositions: List[CompositionNode] = []
        
        # Initialize with ARC primitives
        self._init_arc_primitives()
    
    def _init_arc_primitives(self):
        """Initialize ARC-specific primitives"""
        primitives = [
            ComposablePrimitive("identity", lambda x: x, "grid", "grid", 1),
            ComposablePrimitive("flip_h", lambda x: np.fliplr(x), "grid", "grid", 1),
            ComposablePrimitive("flip_v", lambda x: np.flipud(x), "grid", "grid", 1),
            ComposablePrimitive("rotate_90", lambda x: np.rot90(x), "grid", "grid", 1),
            ComposablePrimitive("rotate_180", lambda x: np.rot90(x, 2), "grid", "grid", 1),
            ComposablePrimitive("invert", lambda x: 9 - x, "grid", "grid", 1),
            ComposablePrimitive("shift_up", lambda x: np.roll(x, -1, axis=0), "grid", "grid", 2),
            ComposablePrimitive("shift_down", lambda x: np.roll(x, 1, axis=0), "grid", "grid", 2),
            ComposablePrimitive("shift_left", lambda x: np.roll(x, -1, axis=1), "grid", "grid", 2),
            ComposablePrimitive("shift_right", lambda x: np.roll(x, 1, axis=1), "grid", "grid", 2),
            ComposablePrimitive("expand", self._expand_grid, "grid", "grid", 3),
            ComposablePrimitive("contract", self._contract_grid, "grid", "grid", 3),
            ComposablePrimitive("transpose", lambda x: x.T, "grid", "grid", 1),
            ComposablePrimitive("color_swap", self._color_swap, "grid", "grid", 2),
            ComposablePrimitive("gravity", self._apply_gravity, "grid", "grid", 4),
            ComposablePrimitive("connect", self._connect_components, "grid", "grid", 5),
        ]
        
        for p in primitives:
            self.primitives[p.name] = p
    
    def _expand_grid(self, grid: np.ndarray, factor: int = 2) -> np.ndarray:
        """Expand grid by factor"""
        from scipy.ndimage import zoom
        return zoom(grid, factor, order=0)
    
    def _contract_grid(self, grid: np.ndarray, factor: int = 2) -> np.ndarray:
        """Contract grid by factor"""
        h, w = grid.shape
        new_h, new_w = h // factor, w // factor
        result = np.zeros((new_h, new_w), dtype=grid.dtype)
        
        for i in range(new_h):
            for j in range(new_w):
                block = grid[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
                # Take most common value
                if len(block.flatten()) > 0:
                    result[i, j] = np.bincount(block.flatten()).argmax()
        
        return result
    
    def _color_swap(self, grid: np.ndarray, mapping: Dict = None) -> np.ndarray:
        """Swap colors"""
        if mapping is None:
            # Default: swap 1 and 2
            mapping = {1: 2, 2: 1}
        
        result = grid.copy()
        for old, new in mapping.items():
            result[grid == old] = new
        return result
    
    def _apply_gravity(self, grid: np.ndarray, direction: str = "down") -> np.ndarray:
        """Apply gravity to objects"""
        result = np.zeros_like(grid)
        
        for col in range(grid.shape[1]):
            column = grid[:, col]
            non_zero = column[column > 0]
            
            if direction == "down":
                result[-len(non_zero):, col] = non_zero
            else:
                result[:len(non_zero), col] = non_zero
        
        return result
    
    def _connect_components(self, grid: np.ndarray, color: int = None) -> np.ndarray:
        """Connect nearby components"""
        from scipy import ndimage
        
        if color is None:
            color = 1
        
        # Find components
        labeled, num = ndimage.label(grid > 0)
        
        result = grid.copy()
        
        # Connect nearby centers
        for i in range(1, num + 1):
            for j in range(i + 1, num + 1):
                mask1 = (labeled == i)
                mask2 = (labeled == j)
                
                if np.any(mask1) and np.any(mask2):
                    center1 = np.mean(np.where(mask1), axis=1).astype(int)
                    center2 = np.mean(np.where(mask2), axis=1).astype(int)
                    
                    # Draw line between centers
                    y1, x1 = center1
                    y2, x2 = center2
                    
                    # Simple line drawing
                    steps = max(abs(y2-y1), abs(x2-x1))
                    if steps > 0:
                        for t in range(steps + 1):
                            y = int(y1 + (y2-y1) * t / steps)
                            x = int(x1 + (x2-x1) * t / steps)
                            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                                if result[y, x] == 0:
                                    result[y, x] = color
        
        return result
    
    def compose(self, primitive_names: List[str], 
               composition_type: CompositionType = CompositionType.SEQUENTIAL,
               examples: List[Tuple] = None) -> Optional[CompositionNode]:
        """
        Compose primitives into solution
        
        Args:
            primitive_names: Names of primitives to compose
            composition_type: How to compose them
            examples: Optional training examples for validation
            
        Returns:
            Composition node or None if invalid
        """
        if not primitive_names:
            return None
        
        # Get primitives
        primitives = [self.primitives.get(name) for name in primitive_names]
        primitives = [p for p in primitives if p is not None]
        
        if not primitives:
            return None
        
        # Build composition tree
        root = CompositionNode(
            primitive=primitives[-1],  # Last as root
            composition_type=composition_type,
            depth=len(primitives) - 1
        )
        
        # Add children
        for p in primitives[:-1]:
            child = CompositionNode(primitive=p, depth=0)
            root.children.append(child)
        
        # Validate with examples if provided
        if examples:
            score = self._validate_composition(root, examples)
            root.score = score
            
            if score < 0.3:
                return None  # Poor composition
            
            # Store successful composition
            if score > 0.7:
                self.successful_compositions.append(root)
        
        return root
    
    def _validate_composition(self, composition: CompositionNode,
                             examples: List[Tuple]) -> float:
        """Validate composition against examples"""
        scores = []
        
        for inp, expected_out in examples:
            try:
                result = composition.evaluate(inp)
                
                # Score: exact match is 1.0, partial match is proportion
                if result.shape == expected_out.shape:
                    match = np.mean(result == expected_out)
                    scores.append(match)
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def search_composition_space(self, examples: List[Tuple],
                                max_depth: int = 5,
                                timeout: int = 60) -> Optional[CompositionNode]:
        """
        Search for composition that solves examples
        
        Uses beam search with learned heuristics
        """
        start_time = time.time()
        
        # Start with single primitives
        candidates = []
        for name, prim in self.primitives.items():
            node = CompositionNode(primitive=prim)
            score = self._validate_composition(node, examples)
            if score > 0:
                candidates.append((score, node))
        
        # Beam search
        for depth in range(1, max_depth):
            if time.time() - start_time > timeout:
                break
            
            # Sort by score and take top-K
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[:self.beam_width]
            
            new_candidates = []
            
            for score, node in candidates:
                if score > 0.95:  # Good enough
                    return node
                
                # Try extending this composition
                for name, prim in self.primitives.items():
                    if prim.can_compose_with(node.primitive):
                        # Create extended composition
                        extended_names = [name] + self._get_composition_names(node)
                        new_node = self.compose(extended_names, examples=examples)
                        
                        if new_node:
                            new_score = new_node.score
                            new_candidates.append((new_score, new_node))
            
            candidates = new_candidates
            
            if not candidates:
                break
        
        # Return best found
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        return None
    
    def _get_composition_names(self, node: CompositionNode) -> List[str]:
        """Get primitive names from composition tree"""
        names = [node.primitive.name]
        for child in node.children:
            names.extend(self._get_composition_names(child))
        return names
    
    def analogical_transfer(self, source_solution: CompositionNode,
                           target_problem: np.ndarray,
                           source_problem: np.ndarray) -> Optional[CompositionNode]:
        """
        Transfer solution from source to target by analogy
        """
        # Check if problems are analogous
        analogy_score = self._compute_analogy(source_problem, target_problem)
        
        if analogy_score < 0.5:
            return None  # Not analogous enough
        
        # Transfer composition structure
        transferred = self._transfer_structure(source_solution)
        
        return transferred
    
    def _compute_analogy(self, problem1: np.ndarray, problem2: np.ndarray) -> float:
        """Compute structural analogy between problems"""
        # Compare basic statistics
        stats1 = self._problem_stats(problem1)
        stats2 = self._problem_stats(problem2)
        
        # Similarity
        similarities = []
        for key in stats1:
            if key in stats2:
                if isinstance(stats1[key], (int, float)):
                    sim = 1.0 - abs(stats1[key] - stats2[key]) / max(stats1[key], 1)
                    similarities.append(max(0, sim))
                else:
                    similarities.append(1.0 if stats1[key] == stats2[key] else 0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _problem_stats(self, grid: np.ndarray) -> Dict:
        """Extract problem statistics"""
        return {
            'size': grid.size,
            'num_colors': len(np.unique(grid)),
            'density': np.mean(grid > 0),
            'aspect_ratio': grid.shape[1] / max(grid.shape[0], 1),
            'symmetry_h': np.all(grid == np.fliplr(grid)),
            'symmetry_v': np.all(grid == np.flipud(grid)),
        }
    
    def _transfer_structure(self, solution: CompositionNode) -> CompositionNode:
        """Transfer composition structure with adaptation"""
        # For now, simple copy
        # In full implementation, would adapt parameters
        return CompositionNode(
            primitive=solution.primitive,
            children=[self._transfer_structure(c) for c in solution.children],
            composition_type=solution.composition_type
        )
    
    def explain_composition(self, composition: CompositionNode) -> str:
        """Generate human-readable explanation"""
        if not composition.children:
            return f"Apply {composition.primitive.name}"
        
        parts = []
        for child in composition.children:
            parts.append(self.explain_composition(child))
        
        if composition.composition_type == CompositionType.SEQUENTIAL:
            return f"First {' then '.join(parts)}, finally apply {composition.primitive.name}"
        
        elif composition.composition_type == CompositionType.HIERARCHICAL:
            return f"Apply {composition.primitive.name} to ({', then '.join(parts)})"
        
        else:
            return f"Combine {', '.join(parts)} with {composition.primitive.name}"
    
    def get_statistics(self) -> Dict:
        """Get composition statistics"""
        return {
            'primitives_available': len(self.primitives),
            'successful_compositions': len(self.successful_compositions),
            'cache_size': len(self.composition_cache),
            'avg_complexity': np.mean([
                c.total_complexity() for c in self.successful_compositions
            ]) if self.successful_compositions else 0,
            'max_achieved_depth': max([
                c.depth for c in self.successful_compositions
            ], default=0)
        }


import time
