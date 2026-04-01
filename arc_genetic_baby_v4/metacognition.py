"""
Metacognition & Belief Revision Module for ARC-AGI-3
Implements self-reflection, paradigm shifts, and fundamental hypothesis revision
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import copy


class BeliefType(Enum):
    """Types of beliefs that can be revised"""
    CAUSAL = auto()      # Causal relationships
    SYMBOLIC = auto()    # Symbolic rules
    SPATIAL = auto()     # Spatial understanding
    TRANSFORMATION = auto()  # Transformation patterns
    TASK_STRUCTURE = auto()    # How task is structured


class RevisionSeverity(Enum):
    """Severity of belief revision"""
    MINOR = auto()       # Parameter adjustment
    MODERATE = auto()    # Rule replacement
    MAJOR = auto()       # Paradigm shift
    REVOLUTIONARY = auto()  # Complete reconstruction


@dataclass
class Belief:
    """A belief that can be tracked and revised"""
    belief_type: BeliefType
    content: Any
    confidence: float = 0.5
    evidence_count: int = 0
    creation_time: int = 0
    last_revision: int = 0
    revision_history: List[Dict] = field(default_factory=list)
    
    def update_confidence(self, success: bool, alpha: float = 0.1):
        """Update confidence with evidence"""
        self.evidence_count += 1
        # Bayesian-like update
        self.confidence = (1 - alpha) * self.confidence + alpha * float(success)
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Paradigm:
    """A paradigm is a coherent set of beliefs about task structure"""
    name: str
    beliefs: Dict[str, Belief]
    core_assumptions: List[str]
    success_rate: float = 0.5
    attempts: int = 0
    
    def is_viable(self, min_success: float = 0.2, min_attempts: int = 5) -> bool:
        """Check if paradigm is still viable"""
        if self.attempts < min_attempts:
            return True
        return self.success_rate >= min_success


class MetacognitiveMonitor:
    """
    Monitors system performance and triggers metacognitive processes
    """
    
    def __init__(self, failure_threshold: int = 5, 
                 confidence_threshold: float = 0.3):
        self.failure_threshold = failure_threshold
        self.confidence_threshold = confidence_threshold
        
        # Performance history
        self.recent_failures: List[Dict] = []
        self.success_streak: int = 0
        self.failure_streak: int = 0
        
        # Belief tracking
        self.belief_violations: Dict[str, List[Dict]] = defaultdict(list)
        
    def record_attempt(self, success: bool, beliefs_used: List[str],
                      state: Dict, action: str, outcome: Dict):
        """Record task attempt for monitoring"""
        
        if success:
            self.success_streak += 1
            self.failure_streak = 0
        else:
            self.failure_streak += 1
            self.success_streak = 0
            
            # Record which beliefs were violated
            violation = {
                'state': copy.deepcopy(state),
                'action': action,
                'expected': outcome.get('expected'),
                'actual': outcome.get('actual'),
                'beliefs_used': beliefs_used.copy(),
            }
            self.recent_failures.append(violation)
            
            # Track by belief
            for belief_id in beliefs_used:
                self.belief_violations[belief_id].append(violation)
    
    def detect_crisis(self) -> Tuple[bool, Optional[str]]:
        """
        Detect if system is in epistemic crisis
        
        Returns:
            (is_crisis, crisis_type)
        """
        # Crisis 1: Too many consecutive failures
        if self.failure_streak >= self.failure_threshold:
            return True, "consecutive_failures"
        
        # Crisis 2: Belief confidence collapse
        for belief_id, violations in self.belief_violations.items():
            if len(violations) >= self.failure_threshold:
                return True, f"belief_collapse_{belief_id}"
        
        # Crisis 3: Paradigm exhaustion (no progress)
        if len(self.recent_failures) >= self.failure_threshold * 2:
            return True, "paradigm_exhaustion"
        
        return False, None
    
    def get_problematic_beliefs(self) -> List[Tuple[str, int, float]]:
        """
        Get beliefs ranked by how problematic they are
        
        Returns:
            List of (belief_id, violation_count, avg_confidence)
        """
        ranked = []
        for belief_id, violations in self.belief_violations.items():
            if violations:
                # Score: more violations = more problematic
                score = len(violations)
                ranked.append((belief_id, len(violations), score))
        
        return sorted(ranked, key=lambda x: x[2], reverse=True)
    
    def clear_history(self):
        """Clear monitoring history after paradigm shift"""
        self.recent_failures = []
        self.failure_streak = 0
        self.belief_violations.clear()


class BeliefRevisionEngine:
    """
    Engine for revising beliefs when they fail
    Implements Kuhn-style paradigm shifts
    """
    
    def __init__(self):
        self.current_paradigm: Optional[Paradigm] = None
        self.past_paradigms: List[Paradigm] = []
        self.revision_count: int = 0
        
    def create_initial_paradigm(self, initial_beliefs: Dict[str, Any]) -> Paradigm:
        """Create initial paradigm from beliefs"""
        beliefs = {}
        for name, content in initial_beliefs.items():
            belief_type = self._infer_belief_type(name, content)
            beliefs[name] = Belief(
                belief_type=belief_type,
                content=content,
                confidence=0.5,
                creation_time=0
            )
        
        paradigm = Paradigm(
            name="initial",
            beliefs=beliefs,
            core_assumptions=["transformations_are_local", "colors_are_meaningful"]
        )
        
        self.current_paradigm = paradigm
        return paradigm
    
    def _infer_belief_type(self, name: str, content: Any) -> BeliefType:
        """Infer belief type from name and content"""
        if 'causal' in name.lower():
            return BeliefType.CAUSAL
        elif 'symbol' in name.lower() or 'rule' in name.lower():
            return BeliefType.SYMBOLIC
        elif 'spatial' in name.lower():
            return BeliefType.SPATIAL
        elif 'transform' in name.lower():
            return BeliefType.TRANSFORMATION
        else:
            return BeliefType.TASK_STRUCTURE
    
    def revise_belief(self, belief_id: str, 
                     severity: RevisionSeverity,
                     new_evidence: Dict) -> Belief:
        """
        Revise a specific belief
        
        Args:
            belief_id: Which belief to revise
            severity: How deep the revision
            new_evidence: Evidence causing revision
            
        Returns:
            Revised belief
        """
        if not self.current_paradigm or belief_id not in self.current_paradigm.beliefs:
            raise ValueError(f"Belief {belief_id} not found")
        
        old_belief = self.current_paradigm.beliefs[belief_id]
        
        # Record revision
        revision_record = {
            'time': self.revision_count,
            'severity': severity.name,
            'old_content': copy.deepcopy(old_belief.content),
            'new_evidence': new_evidence,
            'old_confidence': old_belief.confidence
        }
        
        # Perform revision based on severity
        if severity == RevisionSeverity.MINOR:
            # Just adjust parameters
            new_content = self._minor_revision(old_belief.content, new_evidence)
            
        elif severity == RevisionSeverity.MODERATE:
            # Replace rule but keep type
            new_content = self._moderate_revision(old_belief, new_evidence)
            
        elif severity == RevisionSeverity.MAJOR:
            # Change fundamental structure
            new_content = self._major_revision(old_belief, new_evidence)
            
        else:  # REVOLUTIONARY
            # Complete reconstruction - will trigger paradigm shift
            return self._revolutionary_revision(belief_id, new_evidence)
        
        # Update belief
        old_belief.content = new_content
        old_belief.revision_history.append(revision_record)
        old_belief.last_revision = self.revision_count
        old_belief.confidence = 0.5  # Reset confidence after revision
        
        self.revision_count += 1
        
        return old_belief
    
    def _minor_revision(self, content: Any, evidence: Dict) -> Any:
        """Minor parameter adjustment"""
        if isinstance(content, dict) and 'threshold' in content:
            # Adjust threshold
            content['threshold'] *= 0.9  # Relax constraint
        return content
    
    def _moderate_revision(self, belief: Belief, evidence: Dict) -> Any:
        """Moderate revision: replace rule"""
        # Generate alternative hypothesis
        old_rule = belief.content
        
        # Try negation or alternative
        if isinstance(old_rule, str):
            if 'not ' in old_rule:
                return old_rule.replace('not ', '')
            else:
                return 'not ' + old_rule
        
        return old_rule
    
    def _major_revision(self, belief: Belief, evidence: Dict) -> Any:
        """Major revision: fundamental restructuring"""
        # Question the whole approach
        current_type = belief.belief_type
        
        # Try alternative type
        alternatives = [t for t in BeliefType if t != current_type]
        new_type = np.random.choice(alternatives)
        
        belief.belief_type = new_type
        
        # Generate new content appropriate for new type
        return self._generate_alternative_content(new_type, evidence)
    
    def _revolutionary_revision(self, belief_id: str, evidence: Dict) -> Belief:
        """Revolutionary: trigger paradigm shift"""
        # Mark current paradigm as failed
        if self.current_paradigm:
            self.past_paradigms.append(self.current_paradigm)
        
        # Create new paradigm
        new_paradigm = self._create_new_paradigm(evidence)
        self.current_paradigm = new_paradigm
        
        return new_paradigm.beliefs.get(belief_id)
    
    def _create_new_paradigm(self, triggering_evidence: Dict) -> Paradigm:
        """Create new paradigm after revolutionary shift"""
        # Identify failed assumptions
        failed_assumptions = []
        if self.current_paradigm:
            for assumption in self.current_paradigm.core_assumptions:
                if self._assumption_violated(assumption, triggering_evidence):
                    failed_assumptions.append(assumption)
        
        # Generate new assumptions by negating/altering failed ones
        new_assumptions = []
        for failed in failed_assumptions:
            new_assumptions.extend(self._generate_alternative_assumptions(failed))
        
        # If no failed assumptions, generate creative new ones
        if not new_assumptions:
            new_assumptions = [
                "transformations_are_global",
                "relations_are_key",
                "context_determines_meaning"
            ]
        
        # Create new beliefs based on new assumptions
        new_beliefs = {}
        for i, assumption in enumerate(new_assumptions):
            new_beliefs[f"paradigm_{len(self.past_paradigms)}_belief_{i}"] = Belief(
                belief_type=BeliefType.TASK_STRUCTURE,
                content=assumption,
                confidence=0.3,  # Low confidence initially
                creation_time=self.revision_count
            )
        
        paradigm = Paradigm(
            name=f"paradigm_{len(self.past_paradigms)}",
            beliefs=new_beliefs,
            core_assumptions=new_assumptions,
            success_rate=0.0,
            attempts=0
        )
        
        return paradigm
    
    def _assumption_violated(self, assumption: str, evidence: Dict) -> bool:
        """Check if an assumption was violated by evidence"""
        # Simple heuristic: if assumption name appears in failure context
        state_desc = str(evidence.get('state', {}))
        return assumption.split('_')[0] in state_desc.lower()
    
    def _generate_alternative_assumptions(self, failed_assumption: str) -> List[str]:
        """Generate alternatives to failed assumption"""
        # Negation
        alternatives = [f"not_{failed_assumption}"]
        
        # Qualification
        alternatives.append(f"conditionally_{failed_assumption}")
        
        # Context-dependent version
        alternatives.append(f"contextual_{failed_assumption}")
        
        return alternatives
    
    def _generate_alternative_content(self, belief_type: BeliefType, 
                                     evidence: Dict) -> Any:
        """Generate new belief content for given type"""
        if belief_type == BeliefType.CAUSAL:
            return {"alternative_causal": evidence.get('unexpected_correlation')}
        elif belief_type == BeliefType.SYMBOLIC:
            return ["alternative_symbol_1", "alternative_symbol_2"]
        else:
            return f"alternative_{belief_type.name.lower()}"
    
    def get_current_beliefs(self) -> Dict[str, Belief]:
        """Get current active beliefs"""
        if not self.current_paradigm:
            return {}
        return self.current_paradigm.beliefs
    
    def explain_revision(self, belief_id: str) -> str:
        """Generate explanation of why belief was revised"""
        if not self.current_paradigm or belief_id not in self.current_paradigm.beliefs:
            return f"Belief {belief_id} not found"
        
        belief = self.current_paradigm.beliefs[belief_id]
        
        if not belief.revision_history:
            return f"Belief '{belief_id}' has never been revised"
        
        last_revision = belief.revision_history[-1]
        
        explanation = (
            f"Belief '{belief_id}' was {last_revision['severity']}ly revised at step "
            f"{last_revision['time']}. "
            f"Confidence dropped from {last_revision['old_confidence']:.2f} to "
            f"{belief.confidence:.2f} due to conflicting evidence."
        )
        
        return explanation


class MetacognitionModule:
    """
    Complete metacognition system integrating monitoring and revision
    """
    
    def __init__(self):
        self.monitor = MetacognitiveMonitor()
        self.revision_engine = BeliefRevisionEngine()
        self.learning_history: List[Dict] = []
        
    def initialize(self, initial_beliefs: Dict[str, Any]):
        """Initialize with starting beliefs"""
        self.revision_engine.create_initial_paradigm(initial_beliefs)
    
    def metacognitive_step(self, success: bool, beliefs_used: List[str],
                          state: Dict, action: str, outcome: Dict) -> Dict:
        """
        Execute one metacognitive step
        
        Returns:
            Dict with any revisions made
        """
        # Record attempt
        self.monitor.record_attempt(success, beliefs_used, state, action, outcome)
        
        result = {
            'crisis_detected': False,
            'revisions': [],
            'paradigm_shift': False
        }
        
        # Check for crisis
        is_crisis, crisis_type = self.monitor.detect_crisis()
        
        if is_crisis:
            result['crisis_detected'] = True
            result['crisis_type'] = crisis_type
            
            # Determine severity based on crisis type
            if 'paradigm' in crisis_type:
                severity = RevisionSeverity.REVOLUTIONARY
                result['paradigm_shift'] = True
            elif 'belief_collapse' in crisis_type:
                severity = RevisionSeverity.MAJOR
            elif 'consecutive' in crisis_type:
                severity = RevisionSeverity.MODERATE
            else:
                severity = RevisionSeverity.MINOR
            
            # Get problematic beliefs
            problematic = self.monitor.get_problematic_beliefs()
            
            # Revise most problematic beliefs
            for belief_id, violation_count, _ in problematic[:3]:
                try:
                    revised = self.revision_engine.revise_belief(
                        belief_id, severity, outcome
                    )
                    result['revisions'].append({
                        'belief_id': belief_id,
                        'severity': severity.name,
                        'new_confidence': revised.confidence
                    })
                except Exception as e:
                    result['revisions'].append({
                        'belief_id': belief_id,
                        'error': str(e)
                    })
            
            # Clear monitor after revision
            self.monitor.clear_history()
        
        # Record for history
        self.learning_history.append({
            'step': len(self.learning_history),
            'success': success,
            'crisis': is_crisis,
            'result': result
        })
        
        return result
    
    def question_assumptions(self) -> List[str]:
        """
        Actively question current assumptions
        Human-like: "Wait, am I understanding this wrong?"
        """
        if not self.revision_engine.current_paradigm:
            return []
        
        assumptions = self.revision_engine.current_paradigm.core_assumptions
        
        # Generate questions about each assumption
        questions = []
        for assumption in assumptions:
            questions.append(f"Is it really true that {assumption.replace('_', ' ')}?")
            questions.append(f"What if {assumption.replace('_', ' ')} is only sometimes true?")
            questions.append(f"Could the opposite of {assumption.replace('_', ' ')} be true?")
        
        return questions
    
    def get_epistemic_status(self) -> Dict:
        """Get current epistemic status"""
        paradigm = self.revision_engine.current_paradigm
        
        if not paradigm:
            return {'status': 'no_paradigm'}
        
        return {
            'current_paradigm': paradigm.name,
            'paradigm_attempts': paradigm.attempts,
            'paradigm_success_rate': paradigm.success_rate,
            'num_beliefs': len(paradigm.beliefs),
            'avg_belief_confidence': np.mean([
                b.confidence for b in paradigm.beliefs.values()
            ]),
            'core_assumptions': paradigm.core_assumptions,
            'past_paradigms': len(self.revision_engine.past_paradigms),
            'total_revisions': self.revision_engine.revision_count,
            'crisis_likelihood': 'high' if self.monitor.failure_streak > 3 else 'low'
        }
