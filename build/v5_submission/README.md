# ARC-AGI-3 V5 Submission

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run submission
python main.py --data data/evaluation --output submission.json

# Or use the script directly
python scripts/kaggle_submission_v5.py --data data/evaluation --output submission.json
```

## Architecture

This submission includes all 6 critical gaps:

1. **Causal Discovery Engine** - Pearl's do-calculus for causal reasoning
2. **Symbolic Abstraction Module** - Rule induction and symbolic reasoning
3. **Counterfactual World Model** - "What-if" scenario simulation
4. **Hierarchical Planner** - Multi-level task decomposition
5. **Learned Attention Mechanism** - Selective focus on relevant features
6. **Zero-Shot Meta-Learning** - Fast adaptation to novel tasks

## Expected Performance

- ARC Score: 40-70% (target: 70%+)
- FPS: 200+
- Memory: <1GB
