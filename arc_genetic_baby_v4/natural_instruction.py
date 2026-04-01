"""
Natural Instruction Learning for ARC-AGI-3
Semantic understanding and knowledge grounding
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto


class SemanticConcept(Enum):
    """Semantic concepts that can be grounded"""
    GRAVITY = "objects fall down"
    SYMMETRY = "mirror image"
    CONTAINMENT = "inside vs outside"
    PATH = "line from A to B"
    REPETITION = "pattern repeats"
    PROGRESSION = "increasing/decreasing"
    XOR = "exclusive or"
    OR = "inclusive or"
    AND = "both together"


@dataclass
class SemanticGrounding:
    """Grounding of semantic concept to visual pattern"""
    concept: SemanticConcept
    visual_pattern: str  # Description of visual pattern
    detector_fn: callable = None
    confidence: float = 0.5
    examples: List[Tuple] = field(default_factory=list)


class NaturalInstructionModule:
    """
    Understands natural language descriptions of tasks
    and grounds them to visual patterns
    """
    
    def __init__(self):
        self.semantic_groundings: Dict[SemanticConcept, SemanticGrounding] = {}
        self.text_to_pattern: Dict[str, str] = {}
        self.pattern_to_transform: Dict[str, callable] = {}
        
        self._init_groundings()
    
    def _init_groundings(self):
        """Initialize semantic groundings"""
        # Gravity
        self.semantic_groundings[SemanticConcept.GRAVITY] = SemanticGrounding(
            concept=SemanticConcept.GRAVITY,
            visual_pattern="objects at bottom, empty space above",
            detector_fn=self._detect_gravity_pattern
        )
        
        # Symmetry
        self.semantic_groundings[SemanticConcept.SYMMETRY] = SemanticGrounding(
            concept=SemanticConcept.SYMMETRY,
            visual_pattern="left side mirrors right side",
            detector_fn=self._detect_symmetry_pattern
        )
        
        # Containment
        self.semantic_groundings[SemanticConcept.CONTAINMENT] = SemanticGrounding(
            concept=SemanticConcept.CONTAINMENT,
            visual_pattern="small object surrounded by larger boundary",
            detector_fn=self._detect_containment_pattern
        )
        
        # Path
        self.semantic_groundings[SemanticConcept.PATH] = SemanticGrounding(
            concept=SemanticConcept.PATH,
            visual_pattern="connected line of pixels between points",
            detector_fn=self._detect_path_pattern
        )
        
        # Initialize text mappings
        self.text_to_pattern = {
            "gravity": "objects fall down",
            "fall": "gravity",
            "drop": "gravity",
            "mirror": "symmetry",
            "symmetric": "symmetry",
            "inside": "containment",
            "outside": "containment",
            "contains": "containment",
            "path": "path",
            "line": "path",
            "connect": "path",
            "repeat": "repetition",
            "pattern": "repetition",
            "increase": "progression",
            "decrease": "progression",
            "progression": "progression",
            "xor": "xor",
            "or": "or",
            "and": "and",
        }
    
    def _detect_gravity_pattern(self, grid: np.ndarray) -> bool:
        """Detect if grid shows gravity pattern"""
        # Check if objects are concentrated at bottom
        h, w = grid.shape
        top_half = grid[:h//2, :]
        bottom_half = grid[h//2:, :]
        
        top_objects = np.sum(top_half > 0)
        bottom_objects = np.sum(bottom_half > 0)
        
        return bottom_objects > top_objects * 2
    
    def _detect_symmetry_pattern(self, grid: np.ndarray) -> bool:
        """Detect if grid shows symmetry"""
        return np.all(grid == np.fliplr(grid)) or np.all(grid == np.flipud(grid))
    
    def _detect_containment_pattern(self, grid: np.ndarray) -> bool:
        """Detect containment relationship"""
        from scipy import ndimage
        
        # Find enclosed regions
        background = grid == 0
        labeled, num = ndimage.label(background)
        
        # Check for completely enclosed regions
        for i in range(1, num + 1):
            region = (labeled == i)
            # Check if surrounded
            if np.any(region):
                return True
        
        return False
    
    def _detect_path_pattern(self, grid: np.ndarray) -> bool:
        """Detect path pattern"""
        # Check for linear structures
        for color in np.unique(grid[grid > 0]):
            mask = (grid == color).astype(int)
            
            # Count pixels
            num_pixels = np.sum(mask)
            if num_pixels > 3:
                # Check if roughly linear
                coords = np.argwhere(mask)
                if len(coords) > 1:
                    # Check if aligned
                    x_range = coords[:, 1].max() - coords[:, 1].min()
                    y_range = coords[:, 0].max() - coords[:, 0].min()
                    
                    # If one dimension much larger than other, it's a line
                    if max(x_range, y_range) > 3 * min(x_range, y_range):
                        return True
        
        return False
    
    def parse_instruction(self, instruction: str) -> List[SemanticConcept]:
        """
        Parse natural language instruction
        
        Example: "make objects fall down" → [GRAVITY]
        Example: "create mirror image" → [SYMMETRY]
        """
        instruction_lower = instruction.lower()
        
        concepts_found = []
        
        for word, pattern in self.text_to_pattern.items():
            if word in instruction_lower:
                # Map to semantic concept
                if pattern == "gravity":
                    concepts_found.append(SemanticConcept.GRAVITY)
                elif pattern == "symmetry":
                    concepts_found.append(SemanticConcept.SYMMETRY)
                elif pattern == "containment":
                    concepts_found.append(SemanticConcept.CONTAINMENT)
                elif pattern == "path":
                    concepts_found.append(SemanticConcept.PATH)
                elif pattern == "repetition":
                    concepts_found.append(SemanticConcept.REPETITION)
                elif pattern == "progression":
                    concepts_found.append(SemanticConcept.PROGRESSION)
                elif pattern == "xor":
                    concepts_found.append(SemanticConcept.XOR)
                elif pattern == "or":
                    concepts_found.append(SemanticConcept.OR)
                elif pattern == "and":
                    concepts_found.append(SemanticConcept.AND)
        
        return list(set(concepts_found))  # Remove duplicates
    
    def ground_concept_to_grid(self, concept: SemanticConcept, 
                               grid: np.ndarray) -> Tuple[bool, float]:
        """
        Check if semantic concept is grounded in this grid
        
        Returns:
            (is_present, confidence)
        """
        if concept not in self.semantic_groundings:
            return False, 0.0
        
        grounding = self.semantic_groundings[concept]
        
        if grounding.detector_fn:
            is_present = grounding.detector_fn(grid)
            return is_present, grounding.confidence if is_present else 0.0
        
        return False, 0.0
    
    def instruction_to_transform(self, instruction: str,
                               examples: List[Tuple]) -> Optional[callable]:
        """
        Convert instruction to transformation function
        
        This is the key capability: understanding what to do!
        """
        concepts = self.parse_instruction(instruction)
        
        if not concepts:
            return None
        
        # Determine transform based on concepts
        if SemanticConcept.GRAVITY in concepts:
            return self._apply_gravity_transform
        
        if SemanticConcept.SYMMETRY in concepts:
            return self._apply_symmetry_transform
        
        if SemanticConcept.CONTAINMENT in concepts:
            return self._apply_containment_transform
        
        if SemanticConcept.PATH in concepts:
            return self._apply_path_transform
        
        return None
    
    def _apply_gravity_transform(self, grid: np.ndarray) -> np.ndarray:
        """Apply gravity transformation"""
        result = np.zeros_like(grid)
        
        for col in range(grid.shape[1]):
            column = grid[:, col]
            non_zero = column[column > 0]
            
            # Place at bottom
            result[-len(non_zero):, col] = non_zero
        
        return result
    
    def _apply_symmetry_transform(self, grid: np.ndarray) -> np.ndarray:
        """Apply symmetry transformation"""
        # Mirror horizontally
        return np.fliplr(grid)
    
    def _apply_containment_transform(self, grid: np.ndarray) -> np.ndarray:
        """Apply containment transformation"""
        # Fill enclosed regions
        from scipy import ndimage
        
        background = grid == 0
        labeled, num = ndimage.label(background)
        
        result = grid.copy()
        
        # Fill interior regions
        for i in range(1, num + 1):
            region = (labeled == i)
            
            # Check if region touches border
            border_mask = np.zeros_like(region)
            border_mask[0, :] = border_mask[-1, :] = True
            border_mask[:, 0] = border_mask[:, -1] = True
            
            if not np.any(region & border_mask):
                # Interior region - fill it
                result[region] = 1  # Default fill color
        
        return result
    
    def _apply_path_transform(self, grid: np.ndarray) -> np.ndarray:
        """Apply path transformation"""
        # Connect components with lines
        from scipy import ndimage
        
        result = grid.copy()
        labeled, num = ndimage.label(grid > 0)
        
        if num < 2:
            return result
        
        # Find centers
        centers = []
        for i in range(1, num + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) > 0:
                center = coords.mean(axis=0).astype(int)
                centers.append(center)
        
        # Connect centers
        for i in range(len(centers) - 1):
            y1, x1 = centers[i]
            y2, x2 = centers[i + 1]
            
            # Draw line
            steps = max(abs(y2-y1), abs(x2-x1))
            if steps > 0:
                for t in range(steps + 1):
                    y = int(y1 + (y2-y1) * t / steps)
                    x = int(x1 + (x2-x1) * t / steps)
                    if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                        if result[y, x] == 0:
                            result[y, x] = 5  # Path color
        
        return result
    
    def learn_grounding(self, instruction: str, 
                      examples: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Learn new grounding from examples
        
        Human points to grid and says "make it fall"
        System learns: "fall" → gravity pattern
        """
        # Parse instruction
        concepts = self.parse_instruction(instruction)
        
        if not concepts:
            # Unknown word - need to create new mapping
            return False
        
        # Learn from examples
        for concept in concepts:
            if concept in self.semantic_groundings:
                grounding = self.semantic_groundings[concept]
                
                # Update examples
                for inp, out in examples:
                    # Check if transform was applied
                    if np.any(inp != out):
                        grounding.examples.append((inp, out))
                
                # Increase confidence
                grounding.confidence = min(1.0, grounding.confidence + 0.1)
        
        return True
    
    def explain_transform(self, transform_name: str) -> str:
        """Generate natural language explanation"""
        explanations = {
            "gravity": "Objects fall down to the bottom of the grid",
            "symmetry": "Create a mirror image across the center",
            "containment": "Fill enclosed spaces or extract contents",
            "path": "Draw connecting lines between objects",
            "repetition": "Copy the pattern to fill the grid",
            "progression": "Increase or decrease values systematically",
        }
        
        return explanations.get(transform_name, f"Apply {transform_name} transformation")
    
    def get_statistics(self) -> Dict:
        """Get grounding statistics"""
        return {
            'groundings_learned': len(self.semantic_groundings),
            'text_mappings': len(self.text_to_pattern),
            'avg_confidence': np.mean([
                g.confidence for g in self.semantic_groundings.values()
            ]),
            'groundings_by_confidence': {
                concept.name: g.confidence
                for concept, g in self.semantic_groundings.items()
            }
        }
