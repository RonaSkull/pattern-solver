"""
measure_explosive_growth.py

Script de validacao para medir crescimento explosivo dos Boom Catalysts.

Compara performance com/sem os 3 catalysts implementados.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
from arc_genetic_baby_v4.config import AgentConfig


def generate_synthetic_puzzle(puzzle_type: str, grid_size: int = 20, num_colors: int = 10) -> Dict:
    """Gera puzzle sintetico para teste"""
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    if puzzle_type == 'rotation':
        color = np.random.randint(1, num_colors)
        grid[5:8, 5:10] = color
        expected = 'rotate'
    elif puzzle_type == 'flip':
        color = np.random.randint(1, num_colors)
        grid[5:10, 5:8] = color
        expected = 'flip_h'
    elif puzzle_type == 'color_shift':
        grid[10:15, 10:15] = 3
        expected = 'color_shift'
    else:
        grid[5:10, 5:10] = 1
        expected = 'identity'
    
    return {'grid': grid, 'expected': expected, 'type': puzzle_type}


def evaluate_agent(agent: ARCGeneticBabyV6, 
                   n_puzzles: int = 20,
                   puzzle_types: List[str] = None) -> float:
    """Avalia agente em puzzles sinteticos"""
    if puzzle_types is None:
        puzzle_types = ['rotation', 'flip', 'color_shift']
    
    scores = []
    for _ in range(n_puzzles):
        ptype = np.random.choice(puzzle_types)
        puzzle = generate_synthetic_puzzle(ptype)
        
        actions = ['rotate', 'flip_h', 'color_shift', 'identity']
        result = agent.step(puzzle['grid'], actions)
        
        # Score: 1.0 se acertou, confidence se errou
        if result.action == puzzle['expected']:
            scores.append(1.0)
        else:
            scores.append(result.confidence * 0.3)
    
    return np.mean(scores)


def measure_growth_curve(n_examples_list: List[int] = None,
                         n_runs: int = 3,
                         output_dir: str = "boom_measurement") -> Dict:
    """
    Mede curva de aprendizado com e sem Boom Catalysts.
    
    Returns:
        Dict com resultados e metricas de crescimento
    """
    if n_examples_list is None:
        n_examples_list = [1, 5, 10, 20, 50]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("🚀 MEDICAO DE CRESCIMENTO EXPLOSIVO - BOOM CATALYSTS")
    print("="*70)
    
    results = {
        'with_boom': [],
        'without_boom': [],
        'n_examples_list': n_examples_list
    }
    
    # Testa COM Boom Catalysts
    print("\n📈 Testando COM Boom Catalysts (Curiosidade + Curriculum + Self-Play)...")
    for n_examples in n_examples_list:
        print(f"  N exemplos: {n_examples}")
        
        run_scores = []
        for run in range(n_runs):
            # Cria agente com Boom Catalysts ativados
            config = AgentConfig(grid_size=20, num_colors=10)
            agent = ARCGeneticBabyV6(config)
            
            # Gera dados via self-play + aprende
            for _ in range(n_examples):
                # 50% self-play, 50% puzzles sinteticos
                if np.random.random() < 0.5:
                    # Self-play
                    sp_examples = agent.self_play_engine.generate_episode(
                        skill_level=agent._estimate_skill_level()
                    )
                    for ex in sp_examples:
                        agent.learn(ex.input_grid, ex.action, ex.target_grid,
                                   success=ex.reward > 0.5, reward=ex.reward)
                else:
                    # Puzzle sintetico
                    puzzle = generate_synthetic_puzzle('rotation')
                    result = agent.step(puzzle['grid'], ['rotate', 'flip_h', 'identity'])
                    agent.learn(puzzle['grid'], result.action, puzzle['grid'],
                               success=False, reward=result.confidence)
                
                # Verifica transicoes de fase
                agent.developmental_curriculum.check_phase_transition(agent)
            
            # Avalia
            score = evaluate_agent(agent)
            run_scores.append(score)
        
        results['with_boom'].append({
            'n_examples': n_examples,
            'mean': float(np.mean(run_scores)),
            'std': float(np.std(run_scores)),
            'runs': run_scores
        })
    
    # Testa SEM Boom Catalysts (baseline)
    print("\n📉 Testando SEM Boom Catalysts (baseline)...")
    for n_examples in n_examples_list:
        print(f"  N exemplos: {n_examples}")
        
        run_scores = []
        for run in range(n_runs):
            # Cria agente com Boom Catalysts DESATIVADOS
            config = AgentConfig(grid_size=20, num_colors=10)
            agent = ARCGeneticBabyV6(config)
            
            # Desativa curiosidade (peso 0)
            agent.curiosity_module.curiosity_weight = 0.0
            
            # Apenas puzzles sinteticos (sem self-play)
            for _ in range(n_examples):
                puzzle = generate_synthetic_puzzle('rotation')
                result = agent.step(puzzle['grid'], ['rotate', 'flip_h', 'identity'])
                agent.learn(puzzle['grid'], result.action, puzzle['grid'],
                           success=False, reward=result.confidence)
            
            # Avalia
            score = evaluate_agent(agent)
            run_scores.append(score)
        
        results['without_boom'].append({
            'n_examples': n_examples,
            'mean': float(np.mean(run_scores)),
            'std': float(np.std(run_scores)),
            'runs': run_scores
        })
    
    # Analisa crescimento
    print("\n📊 Analisando tipo de crescimento...")
    
    def fit_growth_model(scores):
        """Fita modelo: linear, logaritmico, exponencial"""
        x = np.array(n_examples_list)
        y = np.array([s['mean'] for s in scores])
        
        # Linear
        linear_fit = np.polyfit(x, y, 1)
        linear_residual = np.sum((y - np.polyval(linear_fit, x))**2)
        
        # Log
        log_x = np.log(x + 1)
        log_fit = np.polyfit(log_x, y, 1)
        log_residual = np.sum((y - np.polyval(log_fit, log_x))**2)
        
        # Exponencial (aproximado)
        exp_fit = np.polyfit(x, np.log(y + 0.1), 1)
        exp_residual = np.sum((y - np.exp(np.polyval(exp_fit, x)))**2)
        
        # Escolhe melhor fit
        residuals = {'linear': linear_residual, 'log': log_residual, 'exp': exp_residual}
        best = min(residuals, key=residuals.get)
        
        return best, residuals
    
    boom_type, boom_residuals = fit_growth_model(results['with_boom'])
    baseline_type, baseline_residuals = fit_growth_model(results['without_boom'])
    
    results['growth_analysis'] = {
        'with_boom': {'type': boom_type, 'residuals': boom_residuals},
        'without_boom': {'type': baseline_type, 'residuals': baseline_residuals}
    }
    
    # Computa ganho
    final_with = results['with_boom'][-1]['mean']
    final_without = results['without_boom'][-1]['mean']
    improvement = final_with - final_without
    
    results['improvement'] = {
        'absolute': float(improvement),
        'relative': float(improvement / max(final_without, 0.01)),
        'percentage': float(improvement / max(final_without, 0.01) * 100)
    }
    
    # Salva resultados
    with open(output_dir / 'growth_measurement.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Gera plots
    generate_growth_plots(results, output_dir)
    
    # Gera relatorio
    generate_report(results, output_dir)
    
    print("\n" + "="*70)
    print("✅ MEDICAO COMPLETA!")
    print(f"📁 Resultados salvos em: {output_dir}")
    print(f"📈 Melhoria: {improvement*100:.1f}%")
    print(f"🚀 Tipo de crescimento: {boom_type} (com Boom)")
    print("="*70)
    
    return results


def generate_growth_plots(results: Dict, output_dir: Path):
    """Gera visualizacoes do crescimento"""
    
    n_examples = results['n_examples_list']
    with_boom = [r['mean'] for r in results['with_boom']]
    without_boom = [r['mean'] for r in results['without_boom']]
    
    # Plot 1: Curva de crescimento
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_examples, with_boom, 'o-', label='Com Boom Catalysts', 
            color='green', linewidth=2, markersize=8)
    ax.plot(n_examples, without_boom, 's-', label='Sem Boom Catalysts',
            color='red', linewidth=2, markersize=8)
    
    ax.set_xlabel('Numero de Exemplos de Treino')
    ax.set_ylabel('Score de Performance')
    ax.set_title('Curva de Aprendizado: Com vs Sem Boom Catalysts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'growth_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Ganhos
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gains = [w - wo for w, wo in zip(with_boom, without_boom)]
    
    ax.bar(range(len(n_examples)), gains, color='blue', alpha=0.6)
    ax.set_xticks(range(len(n_examples)))
    ax.set_xticklabels(n_examples)
    ax.set_xlabel('Numero de Exemplos')
    ax.set_ylabel('Ganho Absoluto')
    ax.set_title('Ganho de Performance dos Boom Catalysts')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boom_gains.png', dpi=150)
    plt.close()
    
    print(f"  📊 Plots gerados: growth_curve.png, boom_gains.png")


def generate_report(results: Dict, output_dir: Path):
    """Gera relatorio de texto"""
    
    report = f"""# Relatorio de Crescimento Explosivo - Boom Catalysts

## Resumo Executivo

- **Melhoria Absoluta:** {results['improvement']['absolute']*100:.1f}%
- **Melhoria Relativa:** {results['improvement']['percentage']:.1f}%
- **Tipo de Crescimento (Com Boom):** {results['growth_analysis']['with_boom']['type']}
- **Tipo de Crescimento (Sem Boom):** {results['growth_analysis']['without_boom']['type']}

## Curva de Aprendizado

| N Exemplos | Com Boom | Sem Boom | Ganho |
|------------|----------|----------|-------|
"""
    
    for i, n_ex in enumerate(results['n_examples_list']):
        with_score = results['with_boom'][i]['mean'] * 100
        without_score = results['without_boom'][i]['mean'] * 100
        gain = with_score - without_score
        
        report += f"| {n_ex} | {with_score:.1f}% | {without_score:.1f}% | +{gain:.1f}% |\n"
    
    report += f"""
## Analise do Crescimento

### Com Boom Catalysts:
- Modelo: {results['growth_analysis']['with_boom']['type']}
- Residuais: {results['growth_analysis']['with_boom']['residuals']}

### Sem Boom Catalysts:
- Modelo: {results['growth_analysis']['without_boom']['type']}
- Residuais: {results['growth_analysis']['without_boom']['residuals']}

## Conclusao

Os Boom Catalysts demonstraram crescimento **{results['growth_analysis']['with_boom']['type']}**,
enquanto o baseline mostrou crescimento **{results['growth_analysis']['without_boom']['type']}**.

Ganho final de **{results['improvement']['percentage']:.1f}%** com os catalysts implementados.

---
*Gerado por: measure_explosive_growth.py*
*Data: {np.datetime64('now')}*
"""
    
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.write(report)
    
    print(f"  📄 Relatorio gerado: REPORT.md")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Mede crescimento explosivo dos Boom Catalysts')
    parser.add_argument('--output', '-o', default='boom_measurement',
                       help='Diretorio de saida')
    parser.add_argument('--examples', '-e', nargs='+', type=int,
                       default=[1, 5, 10, 20, 50],
                       help='Lista de numeros de exemplos para testar')
    parser.add_argument('--runs', '-r', type=int, default=3,
                       help='Numero de runs por configuracao')
    
    args = parser.parse_args()
    
    results = measure_growth_curve(
        n_examples_list=args.examples,
        n_runs=args.runs,
        output_dir=args.output
    )
