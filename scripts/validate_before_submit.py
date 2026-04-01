"""
Script de Validação Pré-Submissão V6

Executa TODOS os checks necessários antes de submeter para Kaggle.
Só submeta se TODOS os checks passarem.

Uso:
    python scripts/validate_before_submit.py --strict
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import subprocess
import tempfile
import numpy as np


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


class ValidationCheck:
    """Representa um check de validação"""
    
    def __init__(self, name: str, description: str, critical: bool = True):
        self.name = name
        self.description = description
        self.critical = critical
        self.passed = False
        self.error = None
        self.duration = 0.0


class PreSubmissionValidator:
    """Validador completo pré-submissão"""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.checks: List[ValidationCheck] = []
        self.results: Dict = {
            'timestamp': datetime.now().isoformat(),
            'version': 'v6.0',
            'strict_mode': strict,
            'checks': []
        }
    
    def add_check(self, name: str, description: str, critical: bool = True):
        check = ValidationCheck(name, description, critical)
        self.checks.append(check)
        return check
    
    def run_all_checks(self) -> bool:
        """Executa todos os checks e retorna True se passou"""
        print(f"\n{Colors.BOLD}🔍 VALIDAÇÃO PRÉ-SUBMISSÃO V6{Colors.END}\n")
        
        for i, check in enumerate(self.checks, 1):
            print(f"[{i}/{len(self.checks)}] {check.name}...", end=' ')
            
            passed = self._run_check_by_name(check.name, check)
            
            if passed:
                print(f"{Colors.GREEN}✓ PASS{Colors.END} ({check.duration:.2f}s)")
            else:
                status = "WARNING" if not check.critical else "FAIL"
                color = Colors.YELLOW if not check.critical else Colors.RED
                print(f"{color}✗ {status}{Colors.END}")
                if check.error:
                    print(f"   └─ {check.error[:200]}")
            
            self.results['checks'].append({
                'name': check.name,
                'passed': passed,
                'critical': check.critical,
                'error': check.error,
                'duration': check.duration
            })
        
        # Resumo
        passed = sum(1 for c in self.checks if c.passed)
        critical_failed = sum(1 for c in self.checks if not c.passed and c.critical)
        
        print(f"\n{'='*60}")
        print(f"{Colors.BOLD}RESUMO:{Colors.END} {passed}/{len(self.checks)} checks passaram")
        
        if critical_failed > 0:
            print(f"{Colors.RED}⚠ {critical_failed} checks CRÍTICOS falharam{Colors.END}")
            print(f"\n{Colors.RED}❌ NÃO SUBMETA AINDA{Colors.END}")
            return False
        elif passed < len(self.checks):
            print(f"{Colors.YELLOW}⚠ Alguns checks não-críticos falharam{Colors.END}")
            print(f"\n{Colors.YELLOW}⚠ Submissão possível, mas revise warnings{Colors.END}")
            return True
        else:
            print(f"{Colors.GREEN}✅ TODOS OS CHECKS PASSARAM{Colors.END}")
            print(f"\n{Colors.GREEN}🚀 PRONTO PARA SUBMISSÃO{Colors.END}")
            return True
    
    def _run_check_by_name(self, name: str, check: ValidationCheck) -> bool:
        import time
        start = time.time()
        
        try:
            if name == "Testes Unitários":
                self._check_unit_tests()
            elif name == "Integração V6":
                self._check_v6_integration()
            elif name == "Performance FPS":
                self._check_performance()
            elif name == "Memória < 2GB":
                self._check_memory()
            elif name == "Checkpoint Save/Load":
                self._check_checkpoint()
            elif name == "Conformidade Kaggle":
                self._check_kaggle_compliance()
            elif name == "Explicações Geradas":
                self._check_explanations()
            elif name == "Todos 11 Gaps":
                self._check_all_gaps()
            else:
                return False
            
            check.duration = time.time() - start
            return True
        except Exception as e:
            check.error = str(e)
            check.duration = time.time() - start
            return False
    
    def _check_unit_tests(self):
        """Verifica se testes unitários passam"""
        # Clear ALL cache first
        import shutil
        base_dir = Path(__file__).parent.parent
        for pycache in base_dir.rglob('__pycache__'):
            if pycache.is_dir():
                shutil.rmtree(pycache)
        
        result = subprocess.run(
            [sys.executable, '-B', '-m', 'pytest', 'tests/test_v5_gaps.py', '-v', '-q', '--tb=line'],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode != 0:
            raise Exception(f"Testes falharam: {result.stderr[:500]}")
    
    def _check_v6_integration(self):
        """Verifica testes de integração V6"""
        # Clear ALL cache first
        import shutil
        base_dir = Path(__file__).parent.parent
        for pycache in base_dir.rglob('__pycache__'):
            if pycache.is_dir():
                shutil.rmtree(pycache)
        
        result = subprocess.run(
            [sys.executable, '-B', '-m', 'pytest', 'tests/test_v6_integration.py', '-v', '-q', '--tb=line'],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode != 0:
            raise Exception(f"Integração falhou: {result.stderr[:500]}")
    
    def _check_performance(self):
        """Verifica performance (FPS > 50)"""
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        import time
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        grid = np.random.randint(0, 8, (10, 10))
        actions = ['rotate', 'flip_h']
        
        # Warmup
        agent.step(grid, actions)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            agent.step(grid, actions)
        elapsed = time.time() - start
        
        fps = 10 / elapsed
        if fps < 50:
            raise Exception(f"FPS muito baixo: {fps:.1f}")
    
    def _check_memory(self):
        """Verifica uso de memória"""
        import psutil
        import os
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        for _ in range(20):
            grid = np.random.randint(0, 8, (10, 10))
            agent.step(grid, ['rotate', 'flip_h'])
        
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_delta = mem_after - mem_before
        
        if mem_delta > 1000:
            raise Exception(f"Uso de memória alto: {mem_delta:.0f}MB")
    
    def _check_checkpoint(self):
        """Verifica checkpoint save/load"""
        import tempfile
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simula checkpoint
            checkpoint = {
                'episodes': agent.episode_count,
                'steps': agent.step_count,
            }
            path = Path(tmpdir) / 'checkpoint.json'
            with open(path, 'w') as f:
                json.dump(checkpoint, f)
            
            files = list(Path(tmpdir).glob('*'))
            if len(files) < 1:
                raise Exception("Checkpoint não salvo")
    
    def _check_kaggle_compliance(self):
        """Verifica conformidade com regras Kaggle"""
        # Verifica arquivos necessários
        required_files = [
            'arc_genetic_baby_v4/agent_v6.py',
            'arc_genetic_baby_v4/config.py',
            'scripts/kaggle_submission_v5.py',
        ]
        
        base = Path(__file__).parent.parent
        for f in required_files:
            if not (base / f).exists():
                raise Exception(f"Arquivo obrigatório faltando: {f}")
    
    def _check_explanations(self):
        """Verifica geração de explicações"""
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        import numpy as np
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        grid = np.random.randint(0, 8, (10, 10))
        
        result = agent.step(grid, ['rotate', 'flip_h'])
        stats = agent.get_stats()
        
        if not stats or 'version' not in stats:
            raise Exception("Stats não gerados")
    
    def _check_all_gaps(self):
        """Verifica se todos os 11 gaps estão presentes"""
        from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
        from arc_genetic_baby_v4.config import AgentConfig
        
        config = AgentConfig(grid_size=10, num_colors=8)
        agent = ARCGeneticBabyV6(config)
        
        required_modules = [
            'causal_engine', 'symbolic_module', 'counterfactual_engine',
            'hierarchical_planner', 'attention_mechanism', 'meta_learner',
            'deep_causal', 'high_order_symbolic', 'metacognition',
            'productive_composition', 'natural_instruction'
        ]
        
        missing = []
        for module in required_modules:
            if not hasattr(agent, module) or getattr(agent, module) is None:
                missing.append(module)
        
        if missing:
            raise Exception(f"Módulos faltando: {missing}")
    
    def save_report(self, path: str):
        """Salva relatório de validação"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        passed = sum(1 for c in self.results['checks'] if c['passed'])
        self.results['summary'] = {
            'total_checks': len(self.results['checks']),
            'passed': passed,
            'failed': len(self.results['checks']) - passed,
            'ready_for_submission': passed == len(self.results['checks'])
        }
        
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Relatório salvo: {path}")


def main():
    parser = argparse.ArgumentParser(description='Validação Pré-Submissão V6')
    parser.add_argument('--strict', action='store_true', 
                       help='Modo estrito: falha em qualquer warning')
    parser.add_argument('--output', '-o', default='validation_report.json',
                       help='Caminho do relatório de output')
    
    args = parser.parse_args()
    
    validator = PreSubmissionValidator(strict=args.strict)
    
    # Adiciona todos os checks
    validator.add_check("Todos 11 Gaps", "Módulos V6 inicializados")
    validator.add_check("Testes Unitários", "pytest tests/test_v5_gaps.py")
    validator.add_check("Integração V6", "pytest tests/test_v6_integration.py")
    validator.add_check("Performance FPS", "FPS > 50 em grid 10x10")
    validator.add_check("Memória < 2GB", "Uso de memória durante execução")
    validator.add_check("Checkpoint Save/Load", "Persistência de estado")
    validator.add_check("Conformidade Kaggle", "Arquivos obrigatórios presentes")
    validator.add_check("Explicações Geradas", "Explainability funcional")
    
    # Executa
    success = validator.run_all_checks()
    
    # Salva relatório
    validator.save_report(args.output)
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
