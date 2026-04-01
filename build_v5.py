"""
Build script for ARC-AGI-3 V5 Kaggle submission
Packages all modules and creates submission archive
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from datetime import datetime


def build_submission():
    """Build V5 submission package"""
    
    print("🔨 Building ARC-AGI-3 V5 Submission...")
    print("=" * 60)
    
    # Files to include
    core_modules = [
        'arc_genetic_baby_v4/__init__.py',
        'arc_genetic_baby_v4/agent.py',
        'arc_genetic_baby_v4/agent_v5.py',
        'arc_genetic_baby_v4/config.py',
        'arc_genetic_baby_v4/perception.py',
        'arc_genetic_baby_v4/active_inference.py',
        'arc_genetic_baby_v4/program_synthesis.py',
        'arc_genetic_baby_v4/analogy.py',
        'arc_genetic_baby_v4/sleep.py',
        'arc_genetic_baby_v4/memory.py',
        'arc_genetic_baby_v4/causal_discovery.py',
        'arc_genetic_baby_v4/symbolic_abstraction.py',
        'arc_genetic_baby_v4/counterfactual.py',
        'arc_genetic_baby_v4/planner.py',
        'arc_genetic_baby_v4/attention.py',
        'arc_genetic_baby_v4/meta_learning.py',
    ]
    
    scripts = [
        'scripts/kaggle_submission_v5.py',
    ]
    
    # Check all files exist
    print("\n📦 Checking files...")
    missing = []
    for f in core_modules + scripts:
        if not Path(f).exists():
            missing.append(f)
            print(f"  ❌ Missing: {f}")
    
    if missing:
        print(f"\n❌ Build failed: {len(missing)} files missing")
        return False
    
    print(f"  ✅ All {len(core_modules + scripts)} files present")
    
    # Create build directory
    build_dir = Path('build/v5_submission')
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy core modules
    print("\n📂 Copying core modules...")
    for f in core_modules:
        src = Path(f)
        dst = build_dir / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  ✓ {f}")
    
    # Copy scripts
    print("\n📂 Copying scripts...")
    for f in scripts:
        src = Path(f)
        dst = build_dir / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  ✓ {f}")
    
    # Create metadata
    print("\n📝 Creating metadata...")
    metadata = {
        'version': '5.0.0',
        'build_date': datetime.now().isoformat(),
        'gaps_implemented': [
            'Causal Discovery Engine',
            'Symbolic Abstraction Module',
            'Counterfactual World Model',
            'Hierarchical Planner',
            'Learned Attention Mechanism',
            'Zero-Shot Meta-Learning'
        ],
        'files': core_modules + scripts,
        'python_version': sys.version,
    }
    
    import json
    with open(build_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ metadata.json")
    
    # Create requirements.txt
    print("\n📋 Creating requirements...")
    requirements = """
numpy>=1.24.0
torch>=2.0.0
networkx>=3.0
scipy>=1.10.0
scikit-learn>=1.3.0
dea>=1.4.0
joblib>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
""".strip()
    
    with open(build_dir / 'requirements.txt', 'w') as f:
        f.write(requirements)
    
    print(f"  ✓ requirements.txt")
    
    # Create main entry point
    print("\n🚀 Creating entry point...")
    entry_point = '''#!/usr/bin/env python3
"""
ARC-AGI-3 V5 Submission Entry Point
"""

import sys
import os

# Add arc_genetic_baby_v4 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'arc_genetic_baby_v4'))

from scripts.kaggle_submission_v5 import main

if __name__ == '__main__':
    main()
'''
    
    with open(build_dir / 'main.py', 'w') as f:
        f.write(entry_point)
    
    print(f"  ✓ main.py")
    
    # Create README
    print("\n📖 Creating README...")
    readme = '''# ARC-AGI-3 V5 Submission

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
'''
    
    with open(build_dir / 'README.md', 'w') as f:
        f.write(readme)
    
    print(f"  ✓ README.md")
    
    # Create zip archive
    print("\n📦 Creating archive...")
    archive_name = f'arc_agi3_v5_submission_{datetime.now():%Y%m%d_%H%M%S}.zip'
    
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in build_dir.rglob('*'):
            if f.is_file():
                arcname = str(f.relative_to(build_dir))
                zf.write(f, arcname)
    
    print(f"  ✓ {archive_name}")
    
    # Get archive size
    archive_size = Path(archive_name).stat().st_size / 1024 / 1024
    print(f"\n📊 Archive size: {archive_size:.1f} MB")
    
    print("\n" + "=" * 60)
    print("✅ Build Complete!")
    print(f"📦 Archive: {archive_name}")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = build_submission()
    sys.exit(0 if success else 1)
