"""
Symbolic Abstraction Module for ARC-AGI-3
Induces symbolic rules from visual examples
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from abc import ABC, abstractmethod
import itertools
import hashlib


class SymbolType(Enum):
    """Types of symbolic terms"""
    COLOR = auto()
    POSITION = auto()
    SIZE = auto()
    SHAPE = auto()
    RELATION = auto()
    TRANSFORM = auto()
    PREDICATE = auto()


@dataclass
class SymbolicTerm:
    """Atomic term in symbolic language"""
    name: str
    symbol_type: SymbolType
    value: Any
    variables: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.variables:
            return f"{self.name}({','.join(self.variables)})"
        return f"{self.name}={self.value}"


@dataclass
class SymbolicRule:
    """Symbolic rule: CONDITION -> ACTION"""
    name: str
    condition: List[SymbolicTerm]
    action: SymbolicTerm
    confidence: float = 0.5
    support: int = 0
    complexity: int = 1
    
    def matches(self, state: Dict) -> bool:
        """Check if rule matches state"""
        for term in self.condition:
            if not self._evaluate_predicate(term, state):
                return False
        return True
    
    def _evaluate_predicate(self, term: SymbolicTerm, state: Dict) -> bool:
        """Evaluate predicate in state context"""
        if term.symbol_type == SymbolType.PREDICATE:
            if term.name == 'color_equals':
                return state.get('object_color') == term.value
            if term.name == 'touches_edge':
                pos = state.get('object_position_x', 0)
                size = state.get('object_size', 1)
                grid_size = state.get('grid_size', 10)
                return pos == 0 or pos + size >= grid_size
        return True
    
    def apply(self, state: Dict) -> Dict:
        """Apply rule action to state"""
        new_state = state.copy()
        
        if self.action.symbol_type == SymbolType.TRANSFORM:
            if self.action.name == 'set_color':
                new_state['object_color'] = self.action.value
        
        return new_state
    
    def score(self) -> float:
        """Rule score: confidence * support / complexity"""
        if self.complexity == 0:
            return 0
        return (self.confidence * self.support) / self.complexity


class RuleInducer(ABC):
    """Interface for rule induction algorithms"""
    
    @abstractmethod
    def induce(self, examples: List[Tuple[Dict, Dict]], 
              max_rules: int = 10) -> List[SymbolicRule]:
        """Induce rules from (input, output) examples"""
        pass


class BottomUpRuleInducer(RuleInducer):
    """Bottom-up rule induction: generalize from specific examples"""
    
    def __init__(self, max_condition_terms: int = 4, min_support: int = 2):
        self.max_condition_terms = max_condition_terms
        self.min_support = min_support
    
    def induce(self, examples: List[Tuple[Dict, Dict]], 
              max_rules: int = 10) -> List[SymbolicRule]:
        """Induce rules from examples"""
        if len(examples) < self.min_support:
            return []
        
        # Extract deltas
        deltas = []
        for inp, out in examples:
            delta = self._compute_delta(inp, out)
            if delta:
                deltas.append((inp, delta))
        
        if not deltas:
            return []
        
        # Generalize to patterns
        patterns = self._generalize_patterns(deltas)
        
        # Convert to rules
        rules = self._patterns_to_rules(patterns, examples)
        
        # Rank and return top-K
        rules.sort(key=lambda r: r.score(), reverse=True)
        return rules[:max_rules]
    
    def _compute_delta(self, inp: Dict, out: Dict) -> Optional[Dict]:
        """Compute differences between input and output"""
        delta = {}
        for key in out:
            if key in inp and inp[key] != out[key]:
                delta[key] = {
                    'old': inp[key],
                    'new': out[key],
                    'change': self._classify_change(inp[key], out[key])
                }
        return delta if delta else None
    
    def _classify_change(self, old: Any, new: Any) -> str:
        """Classify type of change"""
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            if new > old:
                return 'increase'
            elif new < old:
                return 'decrease'
        if old != new:
            return 'replace'
        return 'no_change'
    
    def _generalize_patterns(self, deltas: List[Tuple[Dict, Dict]]) -> List[Dict]:
        """Generalize deltas into patterns"""
        patterns = []
        
        for inp, delta in deltas:
            for key, change in delta.items():
                pattern = {
                    'changed_var': key,
                    'change_type': change['change'],
                    'context': {k: v for k, v in inp.items() if k not in delta},
                    'frequency': 1
                }
                patterns.append(pattern)
        
        # Group identical patterns
        grouped = defaultdict(list)
        for p in patterns:
            key = (p['changed_var'], p['change_type'])
            grouped[key].append(p)
        
        result = []
        for group in grouped.values():
            if len(group) >= self.min_support:
                p = group[0].copy()
                p['frequency'] = len(group)
                result.append(p)
        
        return result
    
    def _patterns_to_rules(self, patterns: List[Dict], 
                          examples: List[Tuple[Dict, Dict]]) -> List[SymbolicRule]:
        """Convert patterns to symbolic rules"""
        rules = []
        
        for p in patterns:
            # Build condition from context
            condition = []
            for key, value in list(p['context'].items())[:self.max_condition_terms]:
                condition.append(SymbolicTerm(
                    name='value_equals',
                    symbol_type=SymbolType.PREDICATE,
                    value=value,
                    variables=[key]
                ))
            
            # Build action
            action = SymbolicTerm(
                name='transform',
                symbol_type=SymbolType.TRANSFORM,
                value=f"{p['changed_var']}_{p['change_type']}",
                variables=[p['changed_var']]
            )
            
            rule = SymbolicRule(
                name=f"rule_{len(rules)}",
                condition=condition,
                action=action,
                confidence=min(1.0, p['frequency'] / len(examples)),
                support=p['frequency'],
                complexity=len(condition)
            )
            
            rules.append(rule)
        
        return rules


class SymbolicAbstractionModule:
    """
    Main symbolic abstraction module
    Extracts high-level symbols and induces rules
    """
    
    def __init__(self, inducer: RuleInducer = None):
        self.inducer = inducer or BottomUpRuleInducer()
        self.rule_library: List[SymbolicRule] = []
        self.primitives = self._define_primitives()
        self._context_cache: Dict[str, List[SymbolicRule]] = {}
    
    def _define_primitives(self) -> Dict[str, Callable]:
        """Define primitive operations for rule composition"""
        return {
            'set_color': lambda g, c: np.full_like(g, c),
            'shift_color': lambda g, d: np.clip(g + d, 0, 15),
            'rotate_90': lambda g: np.rot90(g, k=1),
            'flip_h': lambda g: np.fliplr(g),
            'flip_v': lambda g: np.flipud(g),
        }
    
    def extract_symbols(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract symbolic representation from grid"""
        symbols = {}
        
        # Extract objects
        objects = self._extract_objects(grid)
        symbols['objects'] = [
            {
                'id': i,
                'color': obj['color'],
                'position': obj['centroid'],
                'size': len(obj['pixels']),
            }
            for i, obj in enumerate(objects)
        ]
        
        # Global features
        symbols['background_color'] = self._estimate_background(grid)
        symbols['num_colors'] = len(np.unique(grid))
        symbols['is_symmetric'] = self._check_symmetry(grid)
        symbols['complexity'] = min(5, len(objects) + symbols['num_colors'] // 4)
        
        # Relations
        if len(objects) >= 2:
            symbols['relations'] = self._extract_relations(objects)
        
        return symbols
    
    def induce_rules(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                    max_rules: int = 10) -> List[SymbolicRule]:
        """Induce rules from (input, output) grid pairs"""
        # Convert to symbolic
        symbolic_examples = [
            (self.extract_symbols(inp), self.extract_symbols(out))
            for inp, out in examples
        ]
        
        # Induce rules
        rules = self.inducer.induce(symbolic_examples, max_rules)
        
        # Add to library
        self.rule_library.extend(rules)
        
        return rules
    
    def apply_rules(self, grid: np.ndarray, 
                   context: Dict = None) -> Tuple[np.ndarray, Optional[SymbolicRule]]:
        """Apply relevant rules to grid"""
        symbols = self.extract_symbols(grid)
        if context:
            symbols.update(context)
        
        # Find applicable rules
        context_key = self._hash_context(symbols)
        if context_key not in self._context_cache:
            applicable = [r for r in self.rule_library if r.matches(symbols)]
            self._context_cache[context_key] = applicable
        else:
            applicable = self._context_cache[context_key]
        
        if not applicable:
            return grid, None
        
        # Apply best rule
        best_rule = max(applicable, key=lambda r: r.score())
        new_symbols = best_rule.apply(symbols)
        
        # Convert back to grid (simplified)
        new_grid = self._symbols_to_grid(new_symbols, grid.shape)
        
        return new_grid, best_rule
    
    def compose_rules(self, rules: List[SymbolicRule], 
                     max_depth: int = 3) -> List[SymbolicRule]:
        """Compose simple rules into complex ones"""
        if len(rules) < 2:
            return []
        
        composed = []
        
        for r1, r2 in itertools.combinations(rules, 2):
            # Check compatibility
            if r1.action.name == r2.condition[0].name if r2.condition else False:
                combined = SymbolicRule(
                    name=f"composed_{r1.name}_{r2.name}",
                    condition=r1.condition + r2.condition[1:],
                    action=r2.action,
                    confidence=(r1.confidence * r2.confidence) ** 0.5,
                    support=min(r1.support, r2.support),
                    complexity=r1.complexity + r2.complexity
                )
                composed.append(combined)
        
        return composed
    
    def _extract_objects(self, grid: np.ndarray) -> List[Dict]:
        """Extract connected objects"""
        from scipy import ndimage
        
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
        
        for color in range(16):
            mask = (grid == color) & ~visited
            if not np.any(mask):
                continue
            
            labeled, num = ndimage.label(mask.astype(int), structure=structure)
            
            for obj_id in range(1, num + 1):
                obj_mask = (labeled == obj_id)
                pixels = list(zip(*np.where(obj_mask)))
                
                if len(pixels) >= 2:
                    centroid = np.mean(pixels, axis=0).astype(int)
                    objects.append({
                        'color': color,
                        'pixels': pixels,
                        'centroid': tuple(centroid),
                    })
                    visited[obj_mask] = True
        
        return objects
    
    def _estimate_background(self, grid: np.ndarray) -> int:
        """Estimate background color"""
        border = np.concatenate([grid[0,:], grid[-1,:], grid[:,0], grid[:,-1]])
        return int(np.bincount(border).argmax())
    
    def _check_symmetry(self, grid: np.ndarray) -> bool:
        """Check for symmetry"""
        return np.array_equal(grid, np.fliplr(grid)) or np.array_equal(grid, np.flipud(grid))
    
    def _extract_relations(self, objects: List[Dict]) -> List[Dict]:
        """Extract spatial relations between objects"""
        relations = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                
                pos1, pos2 = obj1['centroid'], obj2['centroid']
                dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                
                direction = 'unknown'
                dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
                if abs(dx) > abs(dy):
                    direction = 'right' if dx > 0 else 'left'
                else:
                    direction = 'below' if dy > 0 else 'above'
                
                relations.append({
                    'obj1': i, 'obj2': j,
                    'distance': dist,
                    'direction': direction
                })
        
        return relations
    
    def _symbols_to_grid(self, symbols: Dict, shape: Tuple) -> np.ndarray:
        """Convert symbolic representation back to grid"""
        grid = np.zeros(shape, dtype=int)
        
        if 'objects' in symbols:
            for obj in symbols['objects']:
                x, y = obj['position']
                color = obj.get('color', 1)
                size = max(1, int(np.sqrt(obj.get('size', 4))))
                
                for dy in range(size):
                    for dx in range(size):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < shape[1] and 0 <= ny < shape[0]:
                            grid[ny, nx] = color
        
        return grid
    
    def _hash_context(self, context: Dict) -> str:
        """Hash context for caching"""
        items = sorted((k, str(v)) for k, v in context.items() 
                      if isinstance(v, (int, str, bool)))
        return hashlib.md5(str(items).encode()).hexdigest()[:16]
    
    def clear_cache(self):
        """Clear rule cache"""
        self._context_cache.clear()
    
    def save_checkpoint(self, path: str):
        """Save rule library"""
        import json
        from pathlib import Path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        rules_data = [
            {
                'name': r.name,
                'condition': [{'name': t.name, 'type': t.symbol_type.name,
                              'value': t.value} for t in r.condition],
                'action': {'name': r.action.name, 'value': r.action.value},
                'confidence': r.confidence,
                'support': r.support,
            }
            for r in self.rule_library
        ]
        
        with open(path / 'symbolic_rules.json', 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load rule library"""
        import json
        from pathlib import Path
        
        with open(Path(path) / 'symbolic_rules.json', 'r') as f:
            rules_data = json.load(f)
        
        self.rule_library = []
        for rd in rules_data:
            condition = [
                SymbolicTerm(
                    name=t['name'],
                    symbol_type=SymbolType[t['type']],
                    value=t['value']
                ) for t in rd['condition']
            ]
            action = SymbolicTerm(
                name=rd['action']['name'],
                symbol_type=SymbolType.TRANSFORM,
                value=rd['action']['value']
            )
            
            rule = SymbolicRule(
                name=rd['name'],
                condition=condition,
                action=action,
                confidence=rd['confidence'],
                support=rd['support']
            )
            self.rule_library.append(rule)
