#!/usr/bin/env python3
"""Test V6 agent initialization"""
import sys
sys.dont_write_bytecode = True

# Clear all cached modules
for mod in list(sys.modules.keys()):
    if 'arc_genetic' in mod:
        del sys.modules[mod]

from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
from arc_genetic_baby_v4.config import AgentConfig

config = AgentConfig(grid_size=10, num_colors=8)
agent = ARCGeneticBabyV6(config)
print('SUCCESS: V6 Agent initialized!')
print(f'Version: {agent.get_stats()["version"]}')
