"""Main Agent: ARC Genetic Baby V4 - 5-Layer Cognitive Architecture.

Integrates all cognitive layers into a unified agent for ARC-AGI-3:
    1. Predictive Perception (hierarchical predictive coding)
    2. Active Inference (Free Energy minimization)
    3. Program Synthesis (evolutionary algorithm synthesis)
    4. Structural Analogy (structure mapping)
    5. Sleep Consolidation + Genetic Ensemble
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import pickle
import logging
import hashlib
from datetime import datetime

from .config import AgentConfig
from .perception import PredictivePerception, BeliefState
from .active_inference import ActiveInferenceAgent
from .program_synthesis import EvolutionaryProgramSynthesizer
from .analogy import StructuralAnalogyEngine
from .memory import RelationalMemory, Problem
from .sleep import SleepConsolidation, GeneticEnsemble

# Setup structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ARCGeneticBabyV4')


@dataclass
class ActionResult:
    """Result of an agent action."""
    action: str
    confidence: float
    reasoning_path: str  # Which cognitive layer decided
    info: Dict[str, Any]


class ARCGeneticBabyV4:
    """
    Complete ARC-AGI-3 agent integrating 5 cognitive layers.
    
    Architecture:
        - Layer 1: Perception generates hierarchical beliefs
        - Layer 2: Active Inference selects actions to minimize Free Energy
        - Layer 3: Program Synthesis evolves solutions when needed
        - Layer 4: Analogy transfers solutions from similar problems
        - Layer 5: Sleep consolidates + Ensemble votes on final action
    
    Based on:
        - Free Energy Principle (Friston)
        - Structure Mapping Theory (Gentner)
        - Genetic Programming (Koza)
        - Memory Consolidation theory
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.config.validate()
        
        # Initialize 5 cognitive layers
        self.perception = PredictivePerception(
            config=self.config.perception,
            grid_size=self.config.grid_size,
            num_colors=self.config.num_colors
        )
        
        self.active_inference = ActiveInferenceAgent(
            config=self.config.active_inference,
            grid_size=self.config.grid_size,
            num_colors=self.config.num_colors
        )
        
        self.program_synthesis = EvolutionaryProgramSynthesizer(
            config=self.config.program_synthesis
        )
        
        self.analogy_engine = StructuralAnalogyEngine(
            config=self.config.analogy
        )
        
        self.sleep = SleepConsolidation(
            config=self.config.sleep
        )
        
        self.ensemble = GeneticEnsemble(
            config=self.config.ensemble
        )
        
        self.memory = RelationalMemory(
            config=self.config.memory
        )
        
        # Development stage
        self.developmental_stage = 0
        self.experience_count = 0
        
        # Performance tracking
        self.episode_history: List[Dict] = []
        self.total_reward = 0.0
        
    def step(self, frame: np.ndarray, available_actions: List[str]) -> ActionResult:
        """
        Execute one step of the cognitive loop.
        
        Decision flow:
            1. Perception -> hierarchical beliefs
            2. Analogy -> check for transferrable solution
            3. Active Inference -> simulate and minimize Free Energy
            4. Program Synthesis -> evolve if high uncertainty
            5. Ensemble -> genetic vote on final action
        
        Args:
            frame: Current observation [grid_size, grid_size]
            available_actions: List of valid actions
            
        Returns:
            ActionResult with action and metadata
        """
        # === Layer 1: Perception ===
        beliefs = self.perception.infer(frame)
        
        # === Layer 4: Analogy ===
        analogy_result = self._try_analogy(frame, available_actions, beliefs)
        if analogy_result and analogy_result.confidence > 0.8:
            return ActionResult(
                action=analogy_result.action,
                confidence=analogy_result.confidence,
                reasoning_path="analogy",
                info={'analogy_mapping': analogy_result.info}
            )
        
        # === Layer 2: Active Inference ===
        ai_action, ai_info = self.active_inference.select_action(
            beliefs, available_actions
        )
        ai_confidence = ai_info['policy_probability']
        
        # === Layer 3: Program Synthesis (if high uncertainty) ===
        program_action = None
        program_confidence = 0.0
        
        if ai_info['expected_free_energy'] > self.config.active_inference.free_energy_threshold:
            # High uncertainty - try program synthesis
            program_action, program_confidence = self._try_program_synthesis(
                frame, available_actions
            )
        
        # === Layer 5: Ensemble Voting ===
        # Combine proposals from all layers
        candidate_actions = []
        
        # Active Inference proposal
        candidate_actions.append((ai_action, ai_confidence, 'active_inference'))
        
        # Analogy proposal (if available but low confidence)
        if analogy_result:
            candidate_actions.append(
                (analogy_result.action, analogy_result.confidence * 0.8, 'analogy')
            )
        
        # Program synthesis proposal
        if program_action:
            candidate_actions.append((program_action, program_confidence, 'program_synthesis'))
        
        # Get ensemble vote
        ensemble_action, ensemble_info = self.ensemble.vote_action(
            frame, [a[0] for a in candidate_actions], beliefs
        )
        
        # Select final action
        # If ensemble matches one of our proposals, use that with combined confidence
        final_action = ensemble_action
        final_confidence = ensemble_info['vote_shares'].get(ensemble_action, 0.5)
        reasoning = 'ensemble'
        
        # Check if ensemble matches a specific layer's proposal
        for action, conf, source in candidate_actions:
            if action == ensemble_action:
                # Boost confidence with ensemble agreement
                final_confidence = max(final_confidence, conf * 0.9)
                reasoning = f"ensemble+{source}"
                break
        
        return ActionResult(
            action=final_action,
            confidence=final_confidence,
            reasoning_path=reasoning,
            info={
                'ensemble_info': ensemble_info,
                'active_inference_info': ai_info,
                'candidate_actions': candidate_actions,
                'beliefs_confidence': beliefs.confidence,
                'developmental_stage': self.developmental_stage
            }
        )
    
    def _try_analogy(self, frame: np.ndarray, available_actions: List[str],
                    beliefs: BeliefState) -> Optional[ActionResult]:
        """Try to find and apply analogical solution."""
        # Retrieve similar problems from memory
        similar_problems = self.memory.retrieve_similar(frame, k=5)
        
        if not similar_problems:
            return None
        
        # Create current problem
        current_problem = Problem(
            id=f"current_{self.experience_count}",
            grid=frame,
            available_actions=available_actions
        )
        
        # Find analogy
        analogy_solution = self.analogy_engine.find_analogy(
            current_problem, similar_problems
        )
        
        if analogy_solution and analogy_solution.confidence > 0.5:
            return ActionResult(
                action=analogy_solution.transferred_action,
                confidence=analogy_solution.confidence,
                reasoning_path="analogy",
                info={'mapping': analogy_solution.mapping}
            )
        
        return None
    
    def _try_program_synthesis(self, frame: np.ndarray, 
                              available_actions: List[str]) -> Tuple[Optional[str], float]:
        """Try to synthesize a program solution."""
        # Get recent examples from memory
        examples = self.memory.get_recent_examples(n=5)
        
        if len(examples) < 2:
            return None, 0.0
        
        # Try to reuse existing program first
        target_shape = frame.shape
        result = self.program_synthesis.try_reuse_program(frame, target_shape)
        
        if result is not None:
            # Got result from reused program
            # Convert to action (simplified)
            return "transform", 0.7
        
        # Try to evolve new program (if we have time/compute budget)
        # In real-time setting, this might be skipped
        if self.experience_count % 10 == 0:  # Periodic evolution
            try:
                program = self.program_synthesis.evolve_solution(
                    examples, 
                    generations=20,  # Reduced for speed
                    pop_size=50,
                    verbose=False
                )
                
                if program and program.fitness < 0.3:
                    # Good program found
                    return "evolved", 0.6
            except Exception:
                pass
        
        return None, 0.0
    
    def learn(self, state: np.ndarray, action: str, next_state: np.ndarray,
              success: bool, reward: float = 0.0):
        """
        Learning update after action execution.
        
        Updates all layers:
            - Perception: Update prediction model
            - Active Inference: Update generative model
            - Memory: Store experience
            - Sleep: Consolidate if needed
        """
        self.experience_count += 1
        self.total_reward += reward
        
        # Update perception
        self.perception.learn(next_state)
        
        # Update active inference model
        self.active_inference.update_beliefs(state, action, next_state, reward)
        
        # Store in memory
        from .sleep import Experience
        experience = Experience(
            state=state,
            action=action,
            next_state=next_state,
            success=success,
            reward=reward,
            timestamp=self.experience_count,
            problem_id=f"ep_{self.experience_count}"
        )
        
        self.memory.store(experience)
        self.sleep.add_experience(experience)
        
        # Update ensemble agent success
        # Find which agent contributed most
        if self.ensemble.agents:
            agent_id = self.experience_count % len(self.ensemble.agents)
            self.ensemble.update_agent_success(agent_id, success)
        
        # Trigger sleep consolidation periodically
        if len(self.memory.unconsolidated) >= self.config.sleep.consolidation_interval:
            schemas = self.sleep.consolidate()
            self.memory.add_schemas(schemas)
            self.developmental_stage += 1
        
        # Periodic checkpoint
        if self.experience_count % self.config.checkpoint_interval == 0:
            self.save_checkpoint()
    
    def update(self, action: str, reward: float = 0.0, done: bool = False):
        """Alias for learn() - API compatibility with integration scripts."""
        # Store last state for learning
        if hasattr(self, '_last_state') and hasattr(self, '_last_grid'):
            success = reward > 0.5
            self.learn(self._last_state, action, self._last_grid, success, reward)
        
        # Track history
        self.history.append({
            'action': action,
            'reward': reward,
            'done': done,
            'timestamp': self.experience_count
        })
    
    def reset(self):
        """Reset agent state for new episode."""
        self.perception.reset()
        self.active_inference.reset()
        
    def save_checkpoint(self, path: str = None):
        """Save agent checkpoint."""
        if path is None:
            path = self.config.checkpoint_dir / f"checkpoint_{self.experience_count}.pkl"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'experience_count': self.experience_count,
            'developmental_stage': self.developmental_stage,
            'total_reward': self.total_reward,
            'memory': self.memory.save_checkpoint(),
            'sleep': self.sleep.save_checkpoint(),
            'program_library': self.program_synthesis.get_program_library(),
            'ensemble_stats': self.ensemble.get_agent_stats()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        return path
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.experience_count = checkpoint.get('experience_count', 0)
        self.developmental_stage = checkpoint.get('developmental_stage', 0)
        self.total_reward = checkpoint.get('total_reward', 0.0)
        
        self.memory.load_checkpoint(checkpoint.get('memory', {}))
        self.sleep.load_checkpoint(checkpoint.get('sleep', {}))
        
        # Note: Program library and ensemble would need special handling
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'experience_count': self.experience_count,
            'developmental_stage': self.developmental_stage,
            'total_reward': self.total_reward,
            'free_energy_current': self.perception.get_free_energy(),
            'memory_stats': {
                'episodic_count': len(self.memory.episodic),
                'unconsolidated_count': len(self.memory.unconsolidated),
                'schema_count': len(self.memory.semantic_schemas)
            },
            'ensemble_stats': self.ensemble.get_agent_stats()
        }
