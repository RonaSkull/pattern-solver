"""
High-Order Symbolic Abstraction for ARC-AGI-3
Creates new concepts and primitives during problem solving
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import itertools
import re


class ConceptType(Enum):
    """Types of concepts that can be created"""
    OBJECT = auto()      # Visual object
    RELATION = auto()    # Spatial relation
    TRANSFORMATION = auto()  # Transform operation
    PATTERN = auto()     # Repeating pattern
    META = auto()        # Concept about concepts


@dataclass
class Concept:
    """A dynamically created concept"""
    name: str
    concept_type: ConceptType
    definition: Dict[str, Any]  # What defines this concept
    examples: List[Any] = field(default_factory=list)
    confidence: float = 0.5
    created_at: int = 0
    usage_count: int = 0
    
    def matches(self, candidate: Any) -> float:
        """Check if candidate matches this concept (0-1)"""
        score = 0.0
        
        # Check each defining feature
        for feature, expected in self.definition.items():
            if feature in candidate:
                actual = candidate[feature]
                if isinstance(expected, (int, float)):
                    # Numeric similarity
                    similarity = 1.0 - abs(expected - actual) / max(abs(expected), 1)
                    score += max(0, similarity)
                else:
                    # Exact match
                    score += 1.0 if expected == actual else 0.0
        
        return score / max(len(self.definition), 1)
    
    def refine(self, new_example: Dict):
        """Refine concept definition with new example"""
        self.examples.append(new_example)
        
        # Generalize definition
        for key in new_example:
            if key in self.definition:
                old_val = self.definition[key]
                new_val = new_example[key]
                
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    # Average for numeric
                    self.definition[key] = (old_val + new_val) / 2
                elif old_val != new_val:
                    # Create disjunction for categorical
                    if isinstance(old_val, list):
                        if new_val not in old_val:
                            self.definition[key].append(new_val)
                    else:
                        self.definition[key] = [old_val, new_val]


class ConceptCreator:
    """
    Creates new concepts during problem solving
    """
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.concept_counter: int = 0
        self.primitive_library: Dict[str, Callable] = {}
        
        # Initialize with basic primitives
        self._init_primitives()
    
    def _init_primitives(self):
        """Initialize basic visual primitives"""
        self.primitive_library = {
            'square': lambda obj: len(obj.get('pixels', [])) == 
                     obj.get('width', 0) * obj.get('height', 0),
            'line': lambda obj: obj.get('width', 0) == 1 or obj.get('height', 0) == 1,
            'L_shape': self._check_l_shape,
            'symmetric': self._check_symmetry,
            'contiguous': lambda obj: obj.get('num_components', 1) == 1,
        }
    
    def _check_l_shape(self, obj: Dict) -> bool:
        """Check if object is L-shaped"""
        pixels = obj.get('pixels', [])
        if len(pixels) < 3:
            return False
        
        # Check if pixels form an L
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        
        # L has one corner with 2 directions
        return len(set(xs)) >= 2 and len(set(ys)) >= 2
    
    def _check_symmetry(self, obj: Dict) -> bool:
        """Check if object has symmetry"""
        pixels = obj.get('pixels', [])
        if not pixels:
            return False
        
        # Check vertical symmetry
        xs = [p[0] for p in pixels]
        min_x, max_x = min(xs), max(xs)
        center = (min_x + max_x) / 2
        
        symmetric_pairs = 0
        for p in pixels:
            mirror_x = 2 * center - p[0]
            if any(abs(mirror_x - other[0]) < 0.1 and p[1] == other[1] 
                  for other in pixels):
                symmetric_pairs += 1
        
        return symmetric_pairs >= len(pixels) * 0.8
    
    def create_concept_from_examples(self, examples: List[Dict],
                                    concept_type: ConceptType = None) -> Concept:
        """
        Induce new concept from examples
        
        This is the key for ARC: creating concepts like
        "this is like a mirror but with rotation"
        """
        if not examples:
            return None
        
        # Infer concept type if not provided
        if concept_type is None:
            concept_type = self._infer_concept_type(examples)
        
        # Generate name
        self.concept_counter += 1
        name = f"concept_{self.concept_counter}_{concept_type.name.lower()}"
        
        # Extract common features
        common_features = self._extract_common_features(examples)
        
        # Create concept
        concept = Concept(
            name=name,
            concept_type=concept_type,
            definition=common_features,
            examples=examples.copy(),
            created_at=self.concept_counter
        )
        
        self.concepts[name] = concept
        
        return concept
    
    def _infer_concept_type(self, examples: List[Dict]) -> ConceptType:
        """Infer what type of concept these examples represent"""
        if not examples:
            return ConceptType.OBJECT
        
        # Check if examples describe transformations
        if 'input' in examples[0] and 'output' in examples[0]:
            return ConceptType.TRANSFORMATION
        
        # Check if examples describe relations
        if all('relation' in ex or 'relative_to' in ex for ex in examples):
            return ConceptType.RELATION
        
        # Check if pattern
        if len(examples) >= 3:
            return ConceptType.PATTERN
        
        return ConceptType.OBJECT
    
    def _extract_common_features(self, examples: List[Dict]) -> Dict:
        """Extract features common across examples"""
        if not examples:
            return {}
        
        # Start with first example
        common = set(examples[0].keys())
        
        # Intersect with other examples
        for ex in examples[1:]:
            common &= set(ex.keys())
        
        # Find values that are similar
        features = {}
        for key in common:
            if key.startswith('_'):
                continue
            
            values = [ex[key] for ex in examples if key in ex]
            
            if not values:
                continue
            
            if all(isinstance(v, (int, float)) for v in values):
                # Numeric feature - use mean
                features[key] = np.mean(values)
            elif all(v == values[0] for v in values):
                # Constant feature
                features[key] = values[0]
            else:
                # Variable feature - list possible values
                features[key] = list(set(values))
        
        return features
    
    def compose_concepts(self, concept_names: List[str],
                        composition_type: str = "conjunction") -> Concept:
        """
        Compose multiple concepts into new concept
        
        Example: "red" + "square" → "red_square"
        Example: "mirror" + "rotation" → "mirror_with_rotation"
        """
        base_concepts = [self.concepts.get(c) for c in concept_names if c in self.concepts]
        
        if not base_concepts:
            return None
        
        # Generate composed name
        composed_name = f"composed_{'_'.join(concept_names)}"
        
        # Compose definitions based on type
        if composition_type == "conjunction":
            # AND of all features
            composed_def = {}
            for concept in base_concepts:
                composed_def.update(concept.definition)
        
        elif composition_type == "disjunction":
            # OR - union of possible values
            composed_def = {}
            for concept in base_concepts:
                for key, val in concept.definition.items():
                    if key in composed_def:
                        if isinstance(composed_def[key], list):
                            if isinstance(val, list):
                                composed_def[key].extend(val)
                            else:
                                composed_def[key].append(val)
                        else:
                            composed_def[key] = [composed_def[key], val]
                    else:
                        composed_def[key] = val
        
        elif composition_type == "sequence":
            # Temporal sequence
            composed_def = {
                'sequence': [c.definition for c in base_concepts],
                'ordered': True
            }
        
        else:
            # Default: conjunction
            composed_def = {}
            for concept in base_concepts:
                composed_def.update(concept.definition)
        
        # Create composed concept
        composed = Concept(
            name=composed_name,
            concept_type=ConceptType.META,
            definition=composed_def,
            confidence=np.mean([c.confidence for c in base_concepts]) * 0.9
        )
        
        self.concepts[composed_name] = composed
        
        return composed
    
    def create_primitive(self, name: str, 
                        detector: Callable,
                        description: str = ""):
        """Add new primitive to library"""
        self.primitive_library[name] = detector
        
        # Also create as concept
        concept = Concept(
            name=name,
            concept_type=ConceptType.OBJECT,
            definition={'_primitive': True, 'description': description}
        )
        self.concepts[name] = concept
    
    def apply_concept(self, concept_name: str, target: Any) -> float:
        """Apply a concept to classify/understand target"""
        if concept_name not in self.concepts:
            return 0.0
        
        concept = self.concepts[concept_name]
        concept.usage_count += 1
        
        return concept.matches(target)
    
    def get_concept_vocabulary(self) -> Dict[str, List[str]]:
        """Get current open vocabulary"""
        by_type = defaultdict(list)
        
        for name, concept in self.concepts.items():
            by_type[concept.concept_type.name].append(name)
        
        return dict(by_type)
    
    def explain_concept(self, concept_name: str) -> str:
        """Generate human-readable explanation of concept"""
        if concept_name not in self.concepts:
            return f"Concept '{concept_name}' not found"
        
        concept = self.concepts[concept_name]
        
        explanation = f"'{concept_name}' is a {concept.concept_type.name.lower()} concept.\n"
        explanation += f"Defined by:\n"
        
        for feature, value in concept.definition.items():
            if not feature.startswith('_'):
                explanation += f"  - {feature}: {value}\n"
        
        explanation += f"Confidence: {concept.confidence:.2f}, Used {concept.usage_count} times"
        
        return explanation


class HighOrderAbstractionModule:
    """
    Main module for high-order symbolic abstraction
    """
    
    def __init__(self):
        self.creator = ConceptCreator()
        self.abstraction_history: List[Dict] = []
        
    def abstract_from_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Concept]:
        """
        Create new abstractions from input-output examples
        
        This is the key capability for solving novel ARC puzzles!
        """
        new_concepts = []
        
        # 1. Extract objects and their transformations
        for inp, out in examples:
            inp_objects = self._extract_objects(inp)
            out_objects = self._extract_objects(out)
            
            # Find correspondences
            correspondences = self._find_correspondences(inp_objects, out_objects)
            
            # Create transformation concepts
            for inp_obj, out_obj, transform in correspondences:
                transform_concept = self.creator.create_concept_from_examples(
                    [{'transform': transform, 'before': inp_obj, 'after': out_obj}],
                    ConceptType.TRANSFORMATION
                )
                if transform_concept:
                    new_concepts.append(transform_concept)
        
        # 2. Create pattern concepts from sequences
        if len(examples) >= 3:
            pattern_concept = self.creator.create_concept_from_examples(
                [{'index': i, 'input': ex[0].tolist(), 'output': ex[1].tolist()} 
                 for i, ex in enumerate(examples)],
                ConceptType.PATTERN
            )
            if pattern_concept:
                new_concepts.append(pattern_concept)
        
        # 3. Try to compose concepts
        if len(new_concepts) >= 2:
            composed = self.creator.compose_concepts(
                [c.name for c in new_concepts[:2]],
                composition_type="conjunction"
            )
            if composed:
                new_concepts.append(composed)
        
        # Record
        self.abstraction_history.append({
            'num_examples': len(examples),
            'concepts_created': [c.name for c in new_concepts]
        })
        
        return new_concepts
    
    def _extract_objects(self, grid: np.ndarray) -> List[Dict]:
        """Extract objects from grid"""
        from scipy import ndimage
        
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for color in np.unique(grid[grid > 0]):
            mask = (grid == color) & ~visited
            if not np.any(mask):
                continue
            
            labeled, num = ndimage.label(mask)
            
            for i in range(1, num + 1):
                obj_mask = (labeled == i)
                pixels = list(zip(*np.where(obj_mask)))
                
                if len(pixels) >= 2:
                    objects.append({
                        'color': int(color),
                        'pixels': pixels,
                        'size': len(pixels),
                        'centroid': np.mean(pixels, axis=0).tolist(),
                        'bbox': [
                            int(min(p[0] for p in pixels)),
                            int(min(p[1] for p in pixels)),
                            int(max(p[0] for p in pixels)),
                            int(max(p[1] for p in pixels))
                        ]
                    })
                    visited[obj_mask] = True
        
        return objects
    
    def _find_correspondences(self, inp_objects: List[Dict], 
                             out_objects: List[Dict]) -> List[Tuple[Dict, Dict, str]]:
        """Find correspondences between input and output objects"""
        correspondences = []
        
        # Simple matching by color and size similarity
        for inp_obj in inp_objects:
            best_match = None
            best_score = -1
            
            for out_obj in out_objects:
                score = 0
                if inp_obj['color'] == out_obj['color']:
                    score += 1
                
                size_diff = abs(inp_obj['size'] - out_obj['size'])
                score += 1.0 / (1 + size_diff)
                
                if score > best_score:
                    best_score = score
                    best_match = out_obj
            
            if best_match and best_score > 0.5:
                # Determine transformation type
                transform = self._classify_transform(inp_obj, best_match)
                correspondences.append((inp_obj, best_match, transform))
        
        return correspondences
    
    def _classify_transform(self, before: Dict, after: Dict) -> str:
        """Classify what transformation occurred"""
        # Check position change
        pos_change = np.linalg.norm(
            np.array(before['centroid']) - np.array(after['centroid'])
        )
        
        # Check size change
        size_change = after['size'] - before['size']
        
        # Check color change
        color_change = before['color'] != after['color']
        
        # Classify
        if color_change and pos_change < 0.5:
            return "color_change"
        elif pos_change > 2 and size_change == 0:
            return "translation"
        elif size_change != 0 and pos_change < 0.5:
            return "scaling"
        elif pos_change > 2 and size_change != 0:
            return "complex_transform"
        else:
            return "identity"
    
    def solve_with_abstraction(self, test_input: np.ndarray, 
                              learned_concepts: List[Concept]) -> np.ndarray:
        """
        Apply learned abstractions to solve test input
        """
        # Extract objects from test
        test_objects = self._extract_objects(test_input)
        
        # Try to match and apply concepts
        output = test_input.copy()
        
        for obj in test_objects:
            # Find matching transformation concept
            best_concept = None
            best_match = 0
            
            for concept in learned_concepts:
                if concept.concept_type == ConceptType.TRANSFORMATION:
                    match_score = concept.matches({'before': obj})
                    if match_score > best_match:
                        best_match = match_score
                        best_concept = concept
            
            if best_concept and best_match > 0.6:
                # Apply transformation
                transform = best_concept.definition.get('transform', 'identity')
                output = self._apply_transform(output, obj, transform)
        
        return output
    
    def _apply_transform(self, grid: np.ndarray, obj: Dict, 
                        transform: str) -> np.ndarray:
        """Apply transformation to object in grid"""
        output = grid.copy()
        
        if transform == 'color_change':
            # Change color
            new_color = (obj['color'] + 1) % 10
            for y, x in obj['pixels']:
                output[y, x] = new_color
        
        elif transform == 'translation':
            # Simple shift (simplified)
            pass
        
        return output
    
    def get_statistics(self) -> Dict:
        """Get abstraction statistics"""
        vocab = self.creator.get_concept_vocabulary()
        
        return {
            'total_concepts': len(self.creator.concepts),
            'by_type': {k: len(v) for k, v in vocab.items()},
            'primitives': len(self.creator.primitive_library),
            'abstraction_episodes': len(self.abstraction_history),
            'concepts_per_episode': np.mean([
                len(a['concepts_created']) for a in self.abstraction_history
            ]) if self.abstraction_history else 0
        }
