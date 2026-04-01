#!/usr/bin/env python3
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
