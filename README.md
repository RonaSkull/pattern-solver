# ARC-AGI-3 Genetic Baby V4

**Neuro-Cognitive Architecture with Active Inference + Program Synthesis**

[![Tests](https://img.shields.io/badge/tests-23/23%20passed-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![ARC Prize](https://img.shields.io/badge/competition-ARC--AGI--3-orange)](https://arcprize.org)

A 5-layer cognitive architecture designed to achieve human-level performance on the ARC-AGI-3 benchmark through Active Inference, evolutionary program synthesis, and structural analogical reasoning.

## 🎯 MVP Status

| Component | Status | Details |
|-----------|--------|---------|
| **Testes Unitários** | ✅ 23/23 PASSANDO | Todas as camadas validadas |
| **Active Inference** | ✅ Implementado | Free Energy minimization funcional |
| **Program Synthesis** | ✅ Implementado | DEAP + evolução de ASTs |
| **Analogia Estrutural** | ✅ Implementado | SME com mapeamento relacional |
| **Sleep Consolidation** | ✅ Implementado | Replay + compressão de schemas |
| **Ensemble Genético** | ✅ Implementado | Voting com diversidade adaptativa |
| **Otimizações Kaggle** | ✅ 9/9 Implementadas | LRU cache, pruning, parallel voting |
| **Conformidade ARC-AGI-3** | ✅ Validada | Input/output specs, offline, open-source |

### 📊 Expectativa de Pontuação (Realista)

| Cenário | Score Estimado | Comparativo |
|---------|----------------|-------------|
| **MVP Atual** | 5-15% | 12-37x melhor que LLMs puros |
| Humanos | ~100% | Baseline |
| GPT-4/Gemini | 0.2-0.4% | LLMs puros |
| Recorde público | 36.08% | Estado da arte |

### 📈 Rota de Evolução

- **Semana 1**: 5-10% → Pipeline validado no leaderboard
- **Semana 2**: 15-25% → Active Inference + Program Synthesis afinados  
- **Semana 3**: 35-50% → Analogia estrutural transferindo soluções
- **Semana 4+**: 60%+ → Ensemble completo + refinamento
- **Meta Grand Prize**: 100% → Breakthrough em generalização zero-shot

---

## 🧠 Architecture Overview

The agent implements five integrated cognitive layers:

### Layer 1: Predictive Perception (Predictive Coding)
- Hierarchical predictive coding with 3 levels
- Level 1: Pixel/color predictions
- Level 2: Spatial pattern detection (symmetry, rotation, shape)
- Level 3: Structural/goal inference
- Minimizes prediction error (Free Energy) through learning

### Layer 2: Active Inference Engine
- Based on Karl Friston's Free Energy Principle
- Actions minimize expected Free Energy (surprise), not maximize reward
- Policy selection via expected Free Energy minimization
- Simulates action consequences using internal generative model

### Layer 3: Evolutionary Program Synthesis
- Uses DEAP (Distributed Evolutionary Algorithms) for genetic programming
- Evolves compositional programs (ASTs) that transform grids
- Primitives: rotate, flip, color_map, gravitate, fill_holes, etc.
- Fitness: accuracy + simplicity (Occam's razor)

### Layer 4: Structural Analogy Engine
- Implements Gentner's Structure-Mapping Theory (SME)
- Finds structural isomorphisms between problems
- Transfers solutions via systematic correspondence
- One-to-one mapping, parallel connectivity, systematicity

### Layer 5: Sleep Consolidation + Genetic Ensemble
- **Sleep Consolidation**: Replay and compression of experiences into schemas
- **Genetic Ensemble**: Multiple agents with different "DNA" vote on actions
- Diversity maintenance prevents overfitting

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/arc-genetic-baby-v4
cd arc-genetic-baby-v4

# Install dependencies
pip install -e ".[dev]"

# Or install from pyproject.toml
pip install -e .
```

## 🚀 Quick Start

```python
from arc_genetic_baby_v4 import ARCGeneticBabyV4, AgentConfig
import numpy as np

# Create agent
config = AgentConfig()
agent = ARCGeneticBabyV4(config)

# Run agent on ARC task
grid = np.random.randint(0, 10, (10, 10))
actions = ["up", "down", "left", "right", "stay"]

result = agent.step(grid, actions)
print(f"Action: {result.action}, Confidence: {result.confidence}")

# Learning update
agent.learn(
    state=grid,
    action=result.action,
    next_state=np.random.randint(0, 10, (10, 10)),
    success=True,
    reward=1.0
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_agent.py::test_active_inference_convergence -v

# With coverage
pytest tests/ --cov=arc_genetic_baby_v4 --cov-report=html
```

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| ARC-AGI-3 Score | 100% | TBD |
| FPS | 1000+ | TBD |
| Generalization | 80%+ | TBD |

## 🔬 Research Foundation

This architecture is based on:

1. **Active Inference / Free Energy Principle**
   - Friston, K. et al. (2017). Active inference: a process theory.
   - Friston, K. (2019). A free energy principle for a particular physics.

2. **Predictive Processing**
   - Clark, A. (2013). Whatever next? Predictive brains, situated agents.
   - Rao, R.P. & Ballard, D.H. (1999). Predictive coding in the visual cortex.

3. **Genetic Programming**
   - Koza, J.R. (1992). Genetic Programming: On the Programming of Computers.
   - DEAP: Distributed Evolutionary Algorithms in Python.

4. **Structure-Mapping Theory**
   - Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
   - Falkenhainer, B., Forbus, K.D., & Gentner, D. (1989). The Structure-Mapping Engine.

5. **Memory Consolidation**
   - Rasch, B., & Born, J. (2013). About sleep's role in memory.
   - McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). Why there are complementary learning systems.

## 🏗️ Project Structure

```
arc_genetic_baby_v4/
├── __init__.py           # Package exports
├── config.py            # Configuration management
├── agent.py             # Main ARCGeneticBabyV4 class
├── perception.py        # Layer 1: Predictive Perception
├── active_inference.py  # Layer 2: Active Inference
├── program_synthesis.py # Layer 3: Evolutionary Program Synthesis
├── analogy.py           # Layer 4: Structural Analogy Engine
├── sleep.py             # Layer 5: Sleep + Ensemble
└── memory.py            # Relational Memory system

tests/
└── test_agent.py        # Comprehensive test suite
```

## ⚙️ Configuration

Customize via `AgentConfig`:

```python
from arc_genetic_baby_v4.config import (
    AgentConfig,
    PerceptionConfig,
    ActiveInferenceConfig,
    ProgramSynthesisConfig,
)

config = AgentConfig(
    grid_size=64,
    num_colors=16,
    perception=PerceptionConfig(num_levels=3),
    active_inference=ActiveInferenceConfig(horizon=3),
    program_synthesis=ProgramSynthesisConfig(population_size=100),
)
```

Save/load configuration:
```python
config.to_yaml("config.yaml")
loaded = AgentConfig.from_yaml("config.yaml")
```

## 🔄 Training Loop

```python
# Episode training
for episode in range(num_episodes):
    agent.reset()
    state = env.reset()
    
    for step in range(max_steps):
        # Action selection
        result = agent.step(state, env.available_actions)
        
        # Execute action
        next_state, reward, done = env.step(result.action)
        
        # Learning
        agent.learn(state, result.action, next_state, reward > 0, reward)
        
        state = next_state
        if done:
            break
    
    # Periodic checkpoint
    if episode % 100 == 0:
        agent.save_checkpoint()
```

## 📝 ARC-AGI-3 Compliance

- ✅ Grid 64×64, 16 colors
- ✅ 5 directional keys + Undo + Click
- ✅ Turn-based, offline reasoning
- ✅ Open-source (required for prize)
- ✅ No internet during evaluation

## 🛠️ Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black arc_genetic_baby_v4/

# Type checking
mypy arc_genetic_baby_v4/

# Linting
flake8 arc_genetic_baby_v4/
```

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please ensure:
- Tests pass (`pytest`)
- Code is formatted (`black`)
- Type hints added (`mypy`)
- Documentation updated

## 📚 Citation

If using this code in research:

```bibtex
@software{arc_genetic_baby_v4,
  title = {ARC-AGI-3 Genetic Baby V4: Neuro-Cognitive Architecture},
  author = {ARC-AGI-3 Team},
  year = {2024},
  url = {https://github.com/yourusername/arc-genetic-baby-v4}
}
```

## 🔗 References

- [ARC Prize](https://arcprize.org/)
- [ARC-AGI-3 Guidelines](https://arcprize.org/arc-agi-3)
- [DEAP Documentation](https://deap.readthedocs.io/)
- [Active Inference](https://www.fil.ion.ucl.ac.uk/spm/doc/)
