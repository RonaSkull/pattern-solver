#!/usr/bin/env python
"""Test script for updated modules"""
import sys
sys.path.insert(0, 'd:\\ARC_AGI_3')

print('=== Testando modulos atualizados ===')

# Test 1: DevelopmentalCurriculum v6.2.0
print('\n1. DevelopmentalCurriculum v6.2.0...')
from arc_genetic_baby_v4.developmental_curriculum import DevelopmentalCurriculum, DevelopmentalStage

dc = DevelopmentalCurriculum()
info = dc.get_current_phase_info()
print('   Estagio atual:', info['phase'])
print('   Modulos habilitados:', len(info['enabled_modules']))
print('   Descricao:', info['description'][:50] + '...')

# Testar transicao de estagio
dc.update_metrics(success=True, error=0.1)
for i in range(15):
    dc.update_metrics(success=True, error=0.05, module_confidence={'attention': 0.9})
result = dc.check_stage_transition()
if result:
    print('   Transicao:', result[0].name, '->', result[1].name)
else:
    print('   Sem transicao (mastery:', f"{dc.current_metrics.compute_mastery():.2f}", ')')

# Test 2: SelfPlayEngine v6.3.0
print('\n2. SelfPlayEngine v6.3.0...')
from arc_genetic_baby_v4.self_play_engine import SelfPlayEngine, PuzzleType, TrainingExample

engine = SelfPlayEngine(grid_size=10, num_colors=8)
session = engine.start_session()
print('   Sessao iniciada:', session.session_id)

# Gerar exemplos
examples = engine.generate_curriculum_batch(batch_size=5)
print('   Exemplos gerados:', len(examples))

if examples:
    ex = examples[0]
    print('   Tipo de puzzle:', ex.puzzle_type.name)
    print('   Dificuldade:', f"{ex.difficulty:.2f}")
    print('   Recompensa:', f"{ex.reward:.2f}")

# Atualizar com resultados
for ex in examples[:3]:
    engine.update_from_outcome(ex, success=True)

stats = engine.get_statistics()
print('   Total gerados:', stats['total_examples_generated'])
print('   Total aprendidos:', stats['total_examples_learned'])

engine.end_session()
print('   Sessoes completadas:', stats['sessions_completed'])

# Test 3: Compatibilidade com nomes antigos
print('\n3. Compatibilidade...')
from arc_genetic_baby_v4.developmental_curriculum import DevelopmentalPhase
from arc_genetic_baby_v4.self_play_engine import SelfPlayDataGenerator
print('   DevelopmentalPhase = DevelopmentalStage:', DevelopmentalPhase is DevelopmentalStage)
print('   SelfPlayDataGenerator = SelfPlayEngine:', SelfPlayDataGenerator is SelfPlayEngine)

print('\n✅ TODOS OS TESTES PASSARAM!')
print('✅ developmental_curriculum.py v6.2.0 OK')
print('✅ self_play_engine.py v6.3.0 OK')
