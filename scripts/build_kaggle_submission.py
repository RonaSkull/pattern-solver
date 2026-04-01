"""Build Kaggle Submission Package.

Creates submission.zip for ARC-AGI-3 competition with:
- Source code
- Requirements
- Submission metadata
- README for Kaggle

Usage:
    python scripts/build_kaggle_submission.py --version v1.0-mvp --validate
"""

import argparse
import json
import zipfile
import shutil
import sys
from pathlib import Path
from datetime import datetime


SUBMISSION_FILES = [
    'arc_genetic_baby_v4/',
    'pyproject.toml',
    'README.md',
    'LICENSE',  # Will create if missing
]

EXCLUDE_PATTERNS = [
    '__pycache__',
    '*.pyc',
    '*.pyo',
    '.pytest_cache',
    '*.egg-info',
    '.git',
    '.env',
    '*.log',
    'checkpoints/',
    'logs/',
]


def create_license():
    """Create CC-BY 4.0 license for prize eligibility."""
    license_text = """Creative Commons Attribution 4.0 International License

Copyright (c) 2026 ARC Genetic Baby V4 Team

This work is licensed under the Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or
send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

This license is required for eligibility for the ARC Prize.
"""
    return license_text


def create_submission_metadata(version: str) -> dict:
    """Create submission metadata for Kaggle."""
    return {
        'version': version,
        'date': datetime.now().isoformat(),
        'agent_name': 'ARC Genetic Baby V4',
        'architecture': '5-Layer Neuro-Cognitive',
        'layers': [
            'Predictive Perception',
            'Active Inference',
            'Program Synthesis',
            'Structural Analogy',
            'Sleep Consolidation + Genetic Ensemble'
        ],
        'references': [
            'Friston, K. (2010). The free-energy principle.',
            'Parr et al. (2022). Active Inference.',
            'Gentner, D. (1983). Structure-Mapping Theory.',
            'Gulwani et al. (2017). Program Synthesis.'
        ],
        'license': 'CC-BY-4.0',
        'python_version': '3.11',
        'kaggle_competition': 'arc-prize-2026-arc-agi-3'
    }


def create_kaggle_readme():
    """Create README specific for Kaggle submission."""
    readme = """# ARC Genetic Baby V4 - Kaggle Submission

## Overview
5-layer neuro-cognitive architecture for ARC-AGI-3 benchmark.

## Architecture
1. **Predictive Perception** - Hierarchical predictive coding
2. **Active Inference** - Free Energy minimization (Friston 2010)
3. **Program Synthesis** - Evolutionary algorithm (DEAP)
4. **Structural Analogy** - Structure-Mapping Engine (Gentner 1983)
5. **Sleep Consolidation** - Memory replay and schema abstraction
6. **Genetic Ensemble** - Multi-agent voting with diversity bonus

## Usage
```python
from arc_genetic_baby_v4 import ARCGeneticBabyV4, AgentConfig

config = AgentConfig()
agent = ARCGeneticBabyV4(config)

# Perceive and act
grid = np.random.randint(0, 16, (64, 64))
actions = ["up", "down", "left", "right"]
result = agent.step(grid, actions)
```

## Performance
- Target: 1000 FPS
- Memory: < 2GB
- Grid: 64x64, 16 colors

## License
CC-BY 4.0 - Required for ARC Prize eligibility.

## References
See submission_metadata.json for complete references.
"""
    return readme


def validate_submission(submission_dir: Path) -> list:
    """Validate submission package."""
    errors = []
    
    # Check required files
    required = [
        'arc_genetic_baby_v4/__init__.py',
        'arc_genetic_baby_v4/agent.py',
        'pyproject.toml',
        'submission_metadata.json',
        'LICENSE',
        'README_KAGGLE.md'
    ]
    
    for file in required:
        if not (submission_dir / file).exists():
            errors.append(f"Missing required file: {file}")
    
    # Check imports work
    try:
        sys.path.insert(0, str(submission_dir))
        from arc_genetic_baby_v4 import ARCGeneticBabyV4
    except Exception as e:
        errors.append(f"Import error: {e}")
    
    # Check size
    total_size = sum(f.stat().st_size for f in submission_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    if size_mb > 500:
        errors.append(f"Package too large: {size_mb:.1f} MB > 500 MB limit")
    
    return errors


def build_submission(version: str, validate: bool = True) -> Path:
    """Build submission package."""
    print(f"📦 Building Kaggle submission v{version}...")
    
    # Create temp directory
    build_dir = Path('build_submission')
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    
    # Copy source files
    src_dir = Path('arc_genetic_baby_v4')
    if src_dir.exists():
        shutil.copytree(src_dir, build_dir / src_dir.name)
    
    # Copy config files
    for file in ['pyproject.toml', 'README.md']:
        src = Path(file)
        if src.exists():
            shutil.copy(src, build_dir / src.name)
    
    # Create license
    license_path = build_dir / 'LICENSE'
    license_path.write_text(create_license())
    
    # Create Kaggle README
    readme_path = build_dir / 'README_KAGGLE.md'
    readme_path.write_text(create_kaggle_readme())
    
    # Create metadata
    metadata = create_submission_metadata(version)
    metadata_path = build_dir / 'submission_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Validate if requested
    if validate:
        print("🔍 Validating submission...")
        errors = validate_submission(build_dir)
        if errors:
            print("❌ Validation failed:")
            for err in errors:
                print(f"   - {err}")
            return None
        else:
            print("   ✓ Validation passed")
    
    # Create zip
    zip_path = Path(f'submission_{version}.zip')
    print(f"   Creating {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in build_dir.rglob('*'):
            if path.is_file():
                # Check exclusions
                skip = any(pattern in str(path) for pattern in EXCLUDE_PATTERNS)
                if not skip:
                    arcname = str(path.relative_to(build_dir))
                    zf.write(path, arcname)
    
    # Get size
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")
    
    # Cleanup
    shutil.rmtree(build_dir)
    
    print(f"✅ Submission ready: {zip_path}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Build Kaggle submission package')
    parser.add_argument('--version', type=str, required=True,
                       help='Version tag (e.g., v1.0-mvp)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate submission before building')
    parser.add_argument('--clean', action='store_true',
                       help='Remove build artifacts after')
    
    args = parser.parse_args()
    
    print(f"🚀 ARC-AGI-3 Kaggle Submission Builder")
    print(f"   Version: {args.version}")
    print()
    
    # Build
    zip_path = build_submission(args.version, validate=args.validate)
    
    if zip_path:
        print()
        print("📋 Next steps:")
        print("   1. Upload to Kaggle:")
        print(f"      kaggle competitions submit -c arc-prize-2026-arc-agi-3 -f {zip_path} -m \"{args.version}\"")
        print("   2. Monitor leaderboard:")
        print("      kaggle competitions leaderboard -c arc-prize-2026-arc-agi-3")
        return 0
    else:
        print("\n❌ Build failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
