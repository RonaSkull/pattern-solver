"""Layer 4: Structural Analogy Engine - Structure Mapping Theory.

Implements Gentner's Structure-Mapping Theory to find structural isomorphisms
between problems and transfer solutions via analogical reasoning.

Key insight: Analogical reasoning maps relations, not attributes.
When solving a new problem, find a structurally similar solved problem
and transfer the solution via systematic structural correspondence.

References:
    - Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
    - Falkenhainer, B., Forbus, K.D., & Gentner, D. (1989). The structure-mapping 
      engine: Algorithm and examples.
"""

from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import networkx as nx
from collections import defaultdict

from .config import AnalogyConfig


class RelationType(Enum):
    """Types of relations in ARC puzzles."""
    SPATIAL = "spatial"      # Above, below, left, right, inside, outside
    COLOR = "color"          # Same-color, different-color
    SHAPE = "shape"          # Same-shape, larger-than, smaller-than
    TRANSFORMATION = "transformation"  # Rotation, reflection, translation
    CAUSAL = "causal"        # Causes, enables, prevents
    PART_WHOLE = "part_whole"  # Part-of, contains
    SEQUENCE = "sequence"    # Before, after


@dataclass
class Relation:
    """A relational predicate between objects."""
    relation_type: RelationType
    predicate: str
    arguments: List[str]  # Object IDs
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.relation_type, self.predicate, tuple(self.arguments)))
    
    def __eq__(self, other):
        return (self.relation_type == other.relation_type and 
                self.predicate == other.predicate and
                self.arguments == other.arguments)


@dataclass
class Object:
    """An object in the relational graph."""
    id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    
    def add_attribute(self, name: str, value: Any):
        self.attributes[name] = value
    
    def add_relation(self, relation: Relation):
        self.relations.append(relation)


@dataclass
class RelationalGraph:
    """
    Graph representation of a problem's relational structure.
    
    Nodes: Objects
    Edges: Relations between objects
    """
    objects: Dict[str, Object] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    nx_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    
    def add_object(self, obj: Object):
        self.objects[obj.id] = obj
        self.nx_graph.add_node(obj.id, **obj.attributes)
    
    def add_relation(self, rel: Relation):
        self.relations.append(rel)
        if len(rel.arguments) >= 2:
            self.nx_graph.add_edge(
                rel.arguments[0], 
                rel.arguments[1],
                predicate=rel.predicate,
                type=rel.relation_type.value,
                confidence=rel.confidence
            )
    
    def get_object_relations(self, obj_id: str) -> List[Relation]:
        """Get all relations involving an object."""
        return [r for r in self.relations if obj_id in r.arguments]
    
    def get_neighbors(self, obj_id: str) -> List[str]:
        """Get neighboring objects in the graph."""
        if obj_id in self.nx_graph:
            return list(self.nx_graph.neighbors(obj_id))
        return []
    
    def structural_signature(self) -> Tuple:
        """
        Generate structural signature for fast comparison.
        
        Returns tuple of (num_objects, num_relations, relation_type_counts).
        """
        type_counts = defaultdict(int)
        for r in self.relations:
            type_counts[r.relation_type] += 1
        
        return (len(self.objects), len(self.relations), tuple(sorted(type_counts.items())))


@dataclass
class AnalogyMapping:
    """
    A mapping between two relational graphs (base -> target).
    
    Represents the correspondence found by structure mapping.
    """
    base_graph: RelationalGraph
    target_graph: RelationalGraph
    correspondence: Dict[str, str]  # base_obj_id -> target_obj_id
    mapped_relations: List[Tuple[Relation, Relation]]  # (base, target)
    structural_evaluation: float  # Match score
    systematicity_score: float  # Preference for systematic (deep) mappings
    
    def get_target_object(self, base_obj_id: str) -> Optional[str]:
        """Get corresponding target object for a base object."""
        return self.correspondence.get(base_obj_id)
    
    def is_valid(self) -> bool:
        """Check if mapping is one-to-one."""
        target_ids = list(self.correspondence.values())
        return len(target_ids) == len(set(target_ids))


@dataclass
class AnalogicalSolution:
    """A solution transferred via analogy."""
    source_problem_id: str
    target_problem_id: str
    mapping: AnalogyMapping
    transferred_action: str
    confidence: float
    validation_result: Optional[bool] = None


class StructureMappingEngine:
    """
    Implementation of the Structure Mapping Engine (SME).
    
    Finds the best structural correspondence between two problems by:
        1. Finding all possible local matches (identical relations)
        2. Filtering for structurally consistent mappings
        3. Evaluating systematicity (depth of relational structure)
        4. Selecting best global mapping
    
    Key principles:
        - One-to-one mapping: Each base element maps to at most one target
        - Parallel connectivity: If A->B maps to A'->B', then relations from 
          A and A' should be consistent
        - Systematicity: Prefer mappings with deep relational structures
    """
    
    def __init__(self, config: AnalogyConfig = None):
        self.config = config or AnalogyConfig()
        self.match_cache: Dict[Tuple, List[AnalogyMapping]] = {}
        
    def find_mapping(self, base: RelationalGraph, target: RelationalGraph,
                     greedy: bool = False) -> Optional[AnalogyMapping]:
        """
        Find the best structure-preserving mapping from base to target.
        
        Args:
            base: Source problem (known solution)
            target: Target problem (to solve)
            greedy: Use greedy algorithm instead of full search
            
        Returns:
            Best AnalogyMapping or None if no good mapping found
        """
        # Check cache
        cache_key = (base.structural_signature(), target.structural_signature())
        if cache_key in self.match_cache:
            candidates = self.match_cache[cache_key]
            if candidates:
                return max(candidates, key=lambda m: m.structural_evaluation)
            return None
        
        # Step 1: Find local matches (identical relations)
        local_matches = self._find_local_matches(base, target)
        
        if len(local_matches) < self.config.min_common_relations:
            return None
        
        # Step 2: Build global mappings from local matches
        if greedy:
            mappings = [self._greedy_mapping(base, target, local_matches)]
        else:
            mappings = self._build_global_mappings(base, target, local_matches)
        
        # Step 3: Evaluate and score mappings
        scored_mappings = []
        for mapping in mappings:
            if mapping and self._validate_mapping(mapping):
                score = self._evaluate_mapping(mapping)
                mapping.structural_evaluation = score
                scored_mappings.append((mapping, score))
        
        if not scored_mappings:
            return None
        
        # Select best mapping
        best_mapping, best_score = max(scored_mappings, key=lambda x: x[1])
        
        # Cache result
        self.match_cache[cache_key] = [m for m, _ in scored_mappings if m.structural_evaluation > 0.5]
        
        return best_mapping if best_score >= self.config.structural_match_threshold else None
    
    def structural_similarity(self, rel_graph1: nx.DiGraph, 
                             rel_graph2: nx.DiGraph) -> float:
        """
        Implementação simplificada do SME (Structure-Mapping Engine) 
        Score baseado em: sistematicidade, mapeamento 1-1, conectividade paralela
        
        Reference: Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
        """
        # Extrair subgrafos relacionais (ignorar atributos superficiais)
        relations1 = {edge[2] for edge in rel_graph1.edges(data='relation') if edge[2]}
        relations2 = {edge[2] for edge in rel_graph2.edges(data='relation') if edge[2]}
        
        if not relations1 or not relations2:
            return 0.0
        
        # Jaccard de relações + isomorfismo de subgrafo
        intersection = len(relations1 & relations2)
        union = len(relations1 | relations2)
        relation_overlap = intersection / union if union > 0 else 0.0
        
        # Penalizar se houver atributos conflitantes no mapeamento
        attribute_conflicts = self._count_attribute_conflicts(rel_graph1, rel_graph2)
        
        return relation_overlap * (1 - 0.5 * attribute_conflicts)
    
    def _count_attribute_conflicts(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
        """Count attribute conflicts between two relational graphs."""
        attrs1 = {n: d for n, d in graph1.nodes(data=True)}
        attrs2 = {n: d for n, d in graph2.nodes(data=True)}
        
        conflicts = 0
        total = max(len(attrs1), len(attrs2), 1)
        
        for node, attr1 in attrs1.items():
            if node in attrs2:
                attr2 = attrs2[node]
                # Check for conflicting attributes
                for key in set(attr1.keys()) & set(attr2.keys()):
                    if attr1[key] != attr2[key] and key != 'position':
                        conflicts += 1
        
        return conflicts / total
    
    def _find_local_matches(self, base: RelationalGraph, 
                           target: RelationalGraph) -> List[Tuple[Relation, Relation]]:
        """
        Find identical relations between base and target.
        
        Returns list of (base_relation, target_relation) pairs.
        """
        matches = []
        
        for base_rel in base.relations:
            for target_rel in target.relations:
                # Check if relations are identical (same type and predicate)
                if (base_rel.relation_type == target_rel.relation_type and
                    base_rel.predicate == target_rel.predicate):
                    matches.append((base_rel, target_rel))
        
        return matches
    
    def _build_global_mappings(self, base: RelationalGraph,
                              target: RelationalGraph,
                              local_matches: List[Tuple[Relation, Relation]]) -> List[AnalogyMapping]:
        """
        Construct global one-to-one mappings from local matches.
        
        Uses constraint satisfaction to find structurally consistent mappings.
        """
        if not local_matches:
            return []
        
        # Build candidate object correspondences
        candidate_map = defaultdict(set)
        
        for base_rel, target_rel in local_matches:
            for base_arg, target_arg in zip(base_rel.arguments, target_rel.arguments):
                candidate_map[base_arg].add(target_arg)
        
        # Build mappings via backtracking
        mappings = []
        
        def backtrack(current_mapping: Dict[str, str], remaining_base_objects: Set[str]):
            if not remaining_base_objects:
                # Complete mapping, create AnalogyMapping
                mapping = self._create_mapping_from_correspondence(
                    base, target, current_mapping, local_matches
                )
                if mapping:
                    mappings.append(mapping)
                return
            
            # Pick next base object
            base_obj = remaining_base_objects.pop()
            
            # Try each candidate target
            for target_obj in candidate_map.get(base_obj, set()):
                # Check one-to-one constraint
                if target_obj not in current_mapping.values():
                    current_mapping[base_obj] = target_obj
                    backtrack(current_mapping, remaining_base_objects.copy())
                    del current_mapping[base_obj]
            
            # Also try not mapping this object
            backtrack(current_mapping, remaining_base_objects.copy())
        
        # Start with base objects that have matches
        base_objects_with_matches = set(candidate_map.keys())
        if base_objects_with_matches:
            backtrack({}, base_objects_with_matches)
        
        return mappings
    
    def _greedy_mapping(self, base: RelationalGraph,
                       target: RelationalGraph,
                       local_matches: List[Tuple[Relation, Relation]]) -> Optional[AnalogyMapping]:
        """
        Greedy algorithm for building mappings (faster but may miss optimal).
        """
        correspondence = {}
        used_targets = set()
        
        # Sort local matches by specificity (fewer candidates first)
        sorted_matches = sorted(local_matches, 
                               key=lambda m: len(set(m[0].arguments)))
        
        for base_rel, target_rel in sorted_matches:
            for base_arg, target_arg in zip(base_rel.arguments, target_rel.arguments):
                if base_arg not in correspondence and target_arg not in used_targets:
                    correspondence[base_arg] = target_arg
                    used_targets.add(target_arg)
        
        return self._create_mapping_from_correspondence(
            base, target, correspondence, local_matches
        )
    
    def _create_mapping_from_correspondence(self, base: RelationalGraph,
                                           target: RelationalGraph,
                                           correspondence: Dict[str, str],
                                           local_matches: List[Tuple[Relation, Relation]]) -> Optional[AnalogyMapping]:
        """Create an AnalogyMapping from a correspondence dictionary."""
        
        # Filter mapped relations
        mapped_rels = []
        for base_rel, target_rel in local_matches:
            # Check if all arguments are mapped consistently
            args_mapped = all(
                correspondence.get(base_arg) == target_arg
                for base_arg, target_arg in zip(base_rel.arguments, target_rel.arguments)
            )
            if args_mapped:
                mapped_rels.append((base_rel, target_rel))
        
        if not mapped_rels:
            return None
        
        # Calculate systematicity score
        systematicity = self._calculate_systematicity(mapped_rels)
        
        return AnalogyMapping(
            base_graph=base,
            target_graph=target,
            correspondence=correspondence,
            mapped_relations=mapped_rels,
            structural_evaluation=0.0,  # Will be filled later
            systematicity_score=systematicity
        )
    
    def _validate_mapping(self, mapping: AnalogyMapping) -> bool:
        """
        Validate that a mapping satisfies structure mapping constraints.
        
        Checks:
            - One-to-one correspondence
            - Parallel connectivity
        """
        # Check one-to-one
        target_ids = list(mapping.correspondence.values())
        if len(target_ids) != len(set(target_ids)):
            return False
        
        # Check parallel connectivity
        for base_rel, target_rel in mapping.mapped_relations:
            # Verify that related objects are also mapped
            for base_arg in base_rel.arguments:
                if base_arg in mapping.correspondence:
                    target_arg = mapping.correspondence[base_arg]
                    if target_arg not in target_rel.arguments:
                        return False
        
        return True
    
    def _evaluate_mapping(self, mapping: AnalogyMapping) -> float:
        """
        Score a mapping based on multiple criteria.
        
        Components:
            - Coverage: Fraction of base elements mapped
            - Systematicity: Depth of relational structure
            - Similarity: Attribute similarity of mapped objects
        """
        if not mapping.correspondence:
            return 0.0
        
        # Coverage score
        coverage = len(mapping.correspondence) / max(len(mapping.base_graph.objects), 1)
        
        # Systematicity score (already calculated)
        systematicity = mapping.systematicity_score
        
        # Relation coverage
        rel_coverage = len(mapping.mapped_relations) / max(len(mapping.base_graph.relations), 1)
        
        # Weighted combination
        score = (0.3 * coverage + 
                0.4 * systematicity + 
                0.3 * rel_coverage)
        
        return score
    
    def _calculate_systematicity(self, mapped_relations: List[Tuple[Relation, Relation]]) -> float:
        """
        Calculate systematicity: preference for mappings with deep relational structure.
        
        Higher score for mappings that include systematically related relations
        (e.g., if A causes B and A' causes B', this is more systematic than isolated matches).
        """
        if not mapped_relations:
            return 0.0
        
        # Count relation chains (higher-order relations)
        # Simplified: just count unique predicates as proxy for systematicity
        unique_predicates = set()
        for base_rel, _ in mapped_relations:
            unique_predicates.add((base_rel.relation_type, base_rel.predicate))
        
        # More unique predicates = more systematic
        return min(len(unique_predicates) / max(len(mapped_relations), 1), 1.0)


class StructuralAnalogyEngine:
    """
    Main interface for analogical reasoning in the ARC agent.
    
    Manages a library of solved problems and uses SME to find and apply
    analogies to new problems.
    """
    
    def __init__(self, config: AnalogyConfig = None):
        self.config = config or AnalogyConfig()
        self.sme = StructureMappingEngine(config)
        
        # Library of solved problems (problem_id -> (graph, solution))
        self.solved_problems: Dict[str, Tuple[RelationalGraph, str]] = {}
        
        # Analogy cache
        self.analogy_cache: Dict[str, List[AnalogicalSolution]] = {}
        
    def add_solved_problem(self, problem_id: str, graph: RelationalGraph, 
                          solution: str):
        """Add a solved problem to the library."""
        self.solved_problems[problem_id] = (graph, solution)
        
    def find_analogy(self, current_problem: 'Problem', 
                    memory: List['Problem']) -> Optional[AnalogicalSolution]:
        """
        Find and apply analogy from memory to current problem.
        
        Args:
            current_problem: The problem to solve
            memory: Library of previously solved problems
            
        Returns:
            AnalogicalSolution if good analogy found, None otherwise
        """
        # Build relational graph for current problem
        current_graph = self._build_graph(current_problem)
        
        # Try each problem in memory
        best_solution = None
        best_score = 0.0
        
        for past_problem in memory:
            if past_problem.solution is None:
                continue
                
            # Build graph for past problem
            past_graph = self._build_graph(past_problem)
            
            # Find structural mapping
            mapping = self.sme.find_mapping(past_graph, current_graph)
            
            if mapping and mapping.structural_evaluation > self.config.structural_match_threshold:
                # Transfer solution via mapping
                transferred_action = self._transfer_solution(
                    past_problem.solution, mapping
                )
                
                # Validate via mental simulation
                confidence = mapping.structural_evaluation * mapping.systematicity_score
                
                solution = AnalogicalSolution(
                    source_problem_id=past_problem.id,
                    target_problem_id=current_problem.id,
                    mapping=mapping,
                    transferred_action=transferred_action,
                    confidence=confidence
                )
                
                if confidence > best_score:
                    best_score = confidence
                    best_solution = solution
        
        return best_solution
    
    def _build_graph(self, problem: 'Problem') -> RelationalGraph:
        """
        Build relational graph representation of a problem.
        
        Extracts objects and relations from the problem's grid.
        """
        graph = RelationalGraph()
        
        # Extract objects from grid
        from scipy import ndimage
        
        grid = problem.grid if hasattr(problem, 'grid') else problem.state
        
        # Find connected components (objects)
        unique_colors = np.unique(grid[grid > 0])
        
        for color in unique_colors:
            mask = (grid == color).astype(int)
            labeled, num = ndimage.label(mask)
            
            for i in range(1, num + 1):
                coords = np.argwhere(labeled == i)
                obj_id = f"obj_{int(color)}_{i}"
                
                obj = Object(
                    id=obj_id,
                    attributes={
                        'color': int(color),
                        'size': len(coords),
                        'centroid': coords.mean(axis=0).tolist(),
                        'bbox': [
                            int(coords[:, 0].min()), int(coords[:, 1].min()),
                            int(coords[:, 0].max()), int(coords[:, 1].max())
                        ]
                    }
                )
                
                graph.add_object(obj)
        
        # Extract spatial relations between objects
        objects_list = list(graph.objects.values())
        for i, obj1 in enumerate(objects_list):
            for obj2 in objects_list[i+1:]:
                centroid1 = np.array(obj1.attributes['centroid'])
                centroid2 = np.array(obj2.attributes['centroid'])
                
                # Add spatial relations
                if centroid1[0] < centroid2[0]:
                    rel = Relation(RelationType.SPATIAL, "above", [obj1.id, obj2.id])
                else:
                    rel = Relation(RelationType.SPATIAL, "above", [obj2.id, obj1.id])
                graph.add_relation(rel)
                
                if centroid1[1] < centroid2[1]:
                    rel = Relation(RelationType.SPATIAL, "left", [obj1.id, obj2.id])
                else:
                    rel = Relation(RelationType.SPATIAL, "left", [obj2.id, obj1.id])
                graph.add_relation(rel)
                
                # Color relations
                if obj1.attributes['color'] == obj2.attributes['color']:
                    rel = Relation(RelationType.COLOR, "same-color", [obj1.id, obj2.id])
                    graph.add_relation(rel)
        
        return graph
    
    def _transfer_solution(self, source_solution: str, 
                          mapping: AnalogyMapping) -> str:
        """
        Transfer a solution from base to target via the analogy mapping.
        
        Adapts the solution to fit the target problem's structure.
        """
        # Parse source solution (simplified)
        # In practice, would need structured solution representation
        
        # For now, return the action as-is (would need proper adaptation)
        return source_solution
    
    def structural_match(self, graph1: RelationalGraph, 
                        graph2: RelationalGraph) -> float:
        """
        Calculate structural similarity score between two graphs.
        
        Returns score in [0, 1] indicating structural isomorphism.
        """
        mapping = self.sme.find_mapping(graph1, graph2)
        if mapping is None:
            return 0.0
        return mapping.structural_evaluation
    
    def get_similar_problems(self, problem: 'Problem', 
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most structurally similar problems in memory.
        
        Returns list of (problem_id, similarity_score) tuples.
        """
        current_graph = self._build_graph(problem)
        
        similarities = []
        for prob_id, (graph, _) in self.solved_problems.items():
            score = self.structural_match(graph, current_graph)
            if score > 0:
                similarities.append((prob_id, score))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
