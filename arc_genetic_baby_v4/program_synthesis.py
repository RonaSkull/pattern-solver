"""Layer 3: Evolutionary Program Synthesis - DEAP + SymPy.

Implements evolutionary program synthesis using genetic programming to evolve
Abstract Syntax Trees (ASTs) that solve ARC puzzles.

Key insight: Instead of learning weights, we evolve compositional programs
that capture the algorithmic structure of transformations.

References:
    - DEAP: Distributed Evolutionary Algorithms in Python
    - Koza, J.R. (1992). Genetic Programming: On the Programming of 
      Computers by Means of Natural Selection.
"""

from typing import List, Tuple, Callable, Dict, Any, Optional, Set
from dataclasses import dataclass
from functools import partial
import numpy as np
import operator
import random
import sys

# Module-level function for DEAP pickling compatibility
def _random_color():
    """Generate random color (0-9) for DEAP ephemeral constant."""
    return random.randint(0, 9)

# Register for pickle
sys.modules[__name__]._random_color = _random_color

# DEAP imports
from deap import base, creator, tools, gp, algorithms

from .config import ProgramSynthesisConfig


class ARCPrimitives:
    """
    Primitive operations for ARC puzzle transformations.
    
    These are the building blocks for evolved programs.
    """
    
    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        """Rotate grid 90 degrees clockwise."""
        return np.rot90(grid, k=-1)
    
    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        """Rotate grid 180 degrees."""
        return np.rot90(grid, k=2)
    
    @staticmethod
    def rotate_270(grid: np.ndarray) -> np.ndarray:
        """Rotate grid 270 degrees clockwise."""
        return np.rot90(grid, k=-3)
    
    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        """Flip grid horizontally."""
        return np.fliplr(grid)
    
    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        """Flip grid vertically."""
        return np.flipud(grid)
    
    @staticmethod
    def transpose(grid: np.ndarray) -> np.ndarray:
        """Transpose grid."""
        return grid.T
    
    @staticmethod
    def color_map(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
        """Map one color to another."""
        result = grid.copy()
        result[grid == old_color] = new_color
        return result
    
    @staticmethod
    def replace_all_colors(grid: np.ndarray, new_color: int) -> np.ndarray:
        """Replace all non-zero colors with a single color."""
        result = grid.copy()
        result[result > 0] = new_color
        return result
    
    @staticmethod
    def gravitate_down(grid: np.ndarray) -> np.ndarray:
        """Let colored cells fall down (gravity)."""
        result = np.zeros_like(grid)
        for col in range(grid.shape[1]):
            column = grid[:, col]
            non_zero = column[column > 0]
            if len(non_zero) > 0:
                result[-len(non_zero):, col] = non_zero
        return result
    
    @staticmethod
    def gravitate_up(grid: np.ndarray) -> np.ndarray:
        """Let colored cells float up (reverse gravity)."""
        result = np.zeros_like(grid)
        for col in range(grid.shape[1]):
            column = grid[:, col]
            non_zero = column[column > 0]
            if len(non_zero) > 0:
                result[:len(non_zero), col] = non_zero
        return result
    
    @staticmethod
    def gravitate_left(grid: np.ndarray) -> np.ndarray:
        """Let colored cells move left."""
        result = np.zeros_like(grid)
        for row in range(grid.shape[0]):
            row_data = grid[row, :]
            non_zero = row_data[row_data > 0]
            if len(non_zero) > 0:
                result[row, :len(non_zero)] = non_zero
        return result
    
    @staticmethod
    def gravitate_right(grid: np.ndarray) -> np.ndarray:
        """Let colored cells move right."""
        result = np.zeros_like(grid)
        for row in range(grid.shape[0]):
            row_data = grid[row, :]
            non_zero = row_data[row_data > 0]
            if len(non_zero) > 0:
                result[row, -len(non_zero):] = non_zero
        return result
    
    @staticmethod
    def fill_holes(grid: np.ndarray, fill_color: int) -> np.ndarray:
        """Fill enclosed spaces (holes) with color."""
        from scipy import ndimage
        
        # Find background (0) regions surrounded by non-zero
        binary = (grid > 0).astype(int)
        filled = ndimage.binary_fill_holes(binary)
        
        result = grid.copy()
        result[(filled == 1) & (grid == 0)] = fill_color
        return result
    
    @staticmethod
    def crop_to_content(grid: np.ndarray) -> np.ndarray:
        """Crop grid to bounding box of non-zero content."""
        rows = np.any(grid > 0, axis=1)
        cols = np.any(grid > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return grid
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return grid[rmin:rmax+1, cmin:cmax+1]
    
    @staticmethod
    def detect_objects(grid: np.ndarray) -> List[Dict]:
        """Detect connected components (objects) in grid."""
        from scipy import ndimage
        
        objects = []
        for color in np.unique(grid[grid > 0]):
            mask = (grid == color).astype(int)
            labeled, num = ndimage.label(mask)
            
            for i in range(1, num + 1):
                coords = np.argwhere(labeled == i)
                if len(coords) > 0:
                    objects.append({
                        'color': int(color),
                        'size': len(coords),
                        'centroid': coords.mean(axis=0).tolist(),
                        'bbox': [
                            int(coords[:, 0].min()),
                            int(coords[:, 1].min()),
                            int(coords[:, 0].max()),
                            int(coords[:, 1].max())
                        ],
                        'coords': coords.tolist()
                    })
        
        return objects
    
    @staticmethod
    def extract_pattern(grid: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """Extract regions matching a pattern."""
        # Simple template matching
        h, w = pattern.shape
        result = np.zeros_like(grid)
        
        for i in range(grid.shape[0] - h + 1):
            for j in range(grid.shape[1] - w + 1):
                patch = grid[i:i+h, j:j+w]
                if np.array_equal(patch, pattern):
                    result[i:i+h, j:j+w] = patch
        
        return result
    
    @staticmethod
    def identity(grid: np.ndarray) -> np.ndarray:
        """Identity function."""
        return grid.copy()
    
    @staticmethod
    def invert_colors(grid: np.ndarray, max_color: int = 9) -> np.ndarray:
        """Invert colors (max_color - color)."""
        result = grid.copy()
        mask = result > 0
        result[mask] = max_color - result[mask]
        return result
    
    @staticmethod
    def duplicate_horizontal(grid: np.ndarray) -> np.ndarray:
        """Duplicate grid horizontally."""
        return np.tile(grid, (1, 2))
    
    @staticmethod
    def duplicate_vertical(grid: np.ndarray) -> np.ndarray:
        """Duplicate grid vertically."""
        return np.tile(grid, (2, 1))


@dataclass
class EvolvedProgram:
    """A synthesized program with metadata."""
    func: Callable[[np.ndarray], np.ndarray]
    tree: Any  # DEAP individual (tree)
    fitness: float
    complexity: float
    generalization_score: float = 0.0
    
    def __call__(self, grid: np.ndarray) -> np.ndarray:
        return self.func(grid)


class EvolutionaryProgramSynthesizer:
    """
    Evolutionary program synthesis using genetic programming.
    
    Evolves compositional programs (ASTs) that transform input grids to
    match target grids, using ARC primitives as building blocks.
    
    Fitness function combines:
        - Accuracy: pixel-wise match with target
        - Complexity: program size (Occam's razor)
        - Generalization: performance on multiple examples
    """
    
    def __init__(self, config: ProgramSynthesisConfig = None):
        self.config = config or ProgramSynthesisConfig()
        self.primitives = ARCPrimitives()
        
        # DEAP setup
        self._setup_deap()
        
        # Hall of Fame: best programs for reuse
        self.hall_of_fame: List[EvolvedProgram] = []
        
        # Statistics
        self.generation_stats: List[Dict] = []
        
    def _setup_deap(self):
        """Initialize DEAP genetic programming framework."""
        # Create primitive set (non-typed for simplicity)
        self.pset = gp.PrimitiveSet("ARC", 1)  # 1 argument: grid
        
        # Add primitives
        self.pset.addPrimitive(self.primitives.rotate_90, 1, name="rot90")
        self.pset.addPrimitive(self.primitives.rotate_180, 1, name="rot180")
        self.pset.addPrimitive(self.primitives.flip_horizontal, 1, name="flip_h")
        self.pset.addPrimitive(self.primitives.flip_vertical, 1, name="flip_v")
        self.pset.addPrimitive(self.primitives.transpose, 1, name="transpose")
        self.pset.addPrimitive(self.primitives.gravitate_down, 1, name="grav_down")
        self.pset.addPrimitive(self.primitives.gravitate_up, 1, name="grav_up")
        self.pset.addPrimitive(self.primitives.gravitate_left, 1, name="grav_left")
        self.pset.addPrimitive(self.primitives.gravitate_right, 1, name="grav_right")
        self.pset.addPrimitive(self.primitives.crop_to_content, 1, name="crop")
        self.pset.addPrimitive(self.primitives.duplicate_horizontal, 1, name="dup_h")
        self.pset.addPrimitive(self.primitives.duplicate_vertical, 1, name="dup_v")
        self.pset.addPrimitive(self.primitives.invert_colors, 1, name="invert")
        
        # Note: color_map requires 3 arguments (grid, c1, c2) - skip for now to avoid complexity
        # self.pset.addPrimitive(color_map_2arg, 3, name="color_map")
        
        # Add ephemeral constants for colors (using module-level function for pickle)
        self.pset.addEphemeralConstant("color", _random_color)
        
        # Rename arguments
        self.pset.renameArguments(ARG0='grid')
        
        # Create DEAP classes
        # Check if already created (DEAP limitation)
        if not hasattr(creator, 'FitnessMin'):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        
        # Toolbox
        self.toolbox = base.Toolbox()
        
        # Tree generation
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, 
                             min_=1, max_=self.config.max_tree_depth // 2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, 
                               self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, 
                               self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("select", tools.selTournament, 
                              tournsize=self.config.tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, 
                              max_=self.config.max_tree_depth // 3)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, 
                               pset=self.pset)
        
    def evolve_solution(self, examples: List[Tuple[np.ndarray, np.ndarray]],
                       generations: int = None, 
                       pop_size: int = None,
                       verbose: bool = True) -> Optional[EvolvedProgram]:
        """
        Evolve a program that solves the given examples.
        
        Args:
            examples: List of (input_grid, output_grid) tuples
            generations: Number of generations (default from config)
            pop_size: Population size (default from config)
            verbose: Print progress
            
        Returns:
            EvolvedProgram or None if evolution fails
        """
        if generations is None:
            generations = self.config.generations
        if pop_size is None:
            pop_size = self.config.population_size
            
        if len(examples) == 0:
            return None
            
        # Register fitness evaluation for these examples
        self.toolbox.register("evaluate", 
                             lambda ind: self._evaluate(ind, examples))
        
        # Create initial population
        pop = self.toolbox.population(n=pop_size)
        
        # Hall of Fame
        hof = tools.HallOfFame(self.config.hall_of_fame_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution
        pop, log = algorithms.eaSimple(
            pop, self.toolbox,
            cxpb=self.config.crossover_prob,
            mutpb=self.config.mutation_prob,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )
        
        # Prune low-confidence hypotheses (below 0.1 threshold for 1000 FPS optimization)
        def prune_hypotheses(population, threshold=0.1):
            """Remove individuals with fitness below threshold to save memory."""
            pruned = [ind for ind in population if ind.fitness.values[0] >= threshold]
            # If all pruned, keep at least top 10%
            if not pruned and population:
                n_keep = max(1, len(population) // 10)
                pruned = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)[:n_keep]
            return pruned if pruned else population
        
        pop = prune_hypotheses(pop)
        
        # Store stats
        self.generation_stats.append({
            'generations': generations,
            'population': pop_size,
            'log': log,
            'hall_of_fame': hof
        })
        
        if len(hof) == 0:
            return None
            
        # Get best program
        best = hof[0]
        
        # Compile to function
        func = gp.compile(expr=best, pset=self.pset)
        
        # Calculate complexity
        complexity = len(best)
        
        # Calculate fitness
        fitness_value = best.fitness.values[0] if best.fitness.values else float('inf')
        
        # Create program object
        program = EvolvedProgram(
            func=func,
            tree=best,
            fitness=fitness_value,
            complexity=complexity
        )
        
        # Add to hall of fame
        self.hall_of_fame.append(program)
        
        return program
    
    def _evaluate(self, individual, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float]:
        """
        Evaluate fitness of an individual program.
        
        Fitness combines:
            - Accuracy: pixel-wise match (lower is better)
            - Complexity: program size penalty (Occam's razor)
        """
        # Compile individual
        try:
            func = gp.compile(expr=individual, pset=self.pset)
        except Exception:
            return (float('inf'),)
        
        total_error = 0.0
        valid_examples = 0
        
        for input_grid, target_grid in examples:
            try:
                # Execute program
                output = func(input_grid)
                
                # Ensure output has same shape
                if output.shape != target_grid.shape:
                    # Try to resize or skip
                    continue
                
                # Safe error calculation (handle NaN/inf)
                if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                    continue  # Skip invalid outputs
                    
                # Calculate pixel-wise error
                error = np.sum(output != target_grid) / target_grid.size
                if np.isnan(error) or np.isinf(error):
                    continue  # Skip invalid errors
                    
                total_error += error
                valid_examples += 1
                
            except Exception:
                # Invalid program
                continue
        
        if valid_examples == 0:
            return (float('inf'),)
        
        # Average error (safe)
        avg_error = total_error / valid_examples
        if np.isnan(avg_error) or np.isinf(avg_error):
            return (float('inf'),)
        
        # Add complexity penalty (Occam's razor)
        complexity_penalty = self.config.parsimony_coefficient * len(individual)
        
        # Combined fitness (lower is better)
        fitness = avg_error + complexity_penalty
        
        return (fitness,)
    
    def try_reuse_program(self, input_grid: np.ndarray, 
                         target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Try to apply a program from Hall of Fame.
        
        Returns transformed grid if successful, None otherwise.
        """
        # Sort by fitness
        candidates = sorted(self.hall_of_fame, key=lambda p: p.fitness)
        
        for program in candidates[:5]:  # Try top 5
            try:
                output = program(input_grid)
                if output.shape == target_shape:
                    return output
            except Exception:
                continue
        
        return None
    
    def get_program_library(self) -> List[EvolvedProgram]:
        """Get all evolved programs (library of reusable solutions)."""
        return self.hall_of_fame.copy()
    
    def analyze_program(self, program: EvolvedProgram) -> Dict[str, Any]:
        """
        Analyze a program for interpretability.
        
        Returns information about what the program does.
        """
        tree = program.tree
        
        # Count primitives used
        primitive_counts = {}
        for node in tree:
            name = str(node)
            primitive_counts[name] = primitive_counts.get(name, 0) + 1
        
        # Estimate depth
        depth = tree.height if hasattr(tree, 'height') else 0
        
        return {
            'complexity': program.complexity,
            'depth': depth,
            'primitives': primitive_counts,
            'fitness': program.fitness,
            'str': str(tree)
        }
    
    def reset(self):
        """Reset synthesizer state."""
        self.hall_of_fame.clear()
        self.generation_stats.clear()
        self._setup_deap()
