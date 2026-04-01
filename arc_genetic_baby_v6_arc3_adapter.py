"""
arc_genetic_baby_v6_arc3_adapter.py

Adaptador do ARC Genetic Baby V6 para interface ARC-AGI-3
Conecta nosso agente V6 à competição oficial ARC Prize 2026
"""

import sys
import os
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Tentar importar SDK oficial (se disponível)
try:
    from arc_agi import Agent, FrameData, GameAction
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    # Criar stubs para desenvolvimento
    class Agent:
        pass
    
    @dataclass
    class FrameData:
        grid: list
        available_actions: list
        levels_completed: int = 0
        win_levels: int = 0
        step_count: int = 0
        elapsed_time: float = 0.0
        
    class GameAction:
        ACTION0 = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

# Importar nosso agente V6
try:
    from arc_genetic_baby_v4.agent_v6 import ARCGeneticBabyV6
    from arc_genetic_baby_v4.config import AgentConfig
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("⚠️  Agente V6 não encontrado. Verifique o caminho.")


class ARCGeneticBabyV6Adapter(Agent if SDK_AVAILABLE else object):
    """
    Adaptador do Genetic Baby V6 para ARC-AGI-3
    
    Converte FrameData da competição para entrada do V6
    e mapeia saída do V6 para GameAction da competição
    """
    
    def __init__(self, 
                 use_curiosity: bool = True,
                 use_developmental: bool = True,
                 use_self_play: bool = True,
                 verbose: bool = True):
        self.verbose = verbose
        self.use_curiosity = use_curiosity
        self.use_developmental = use_developmental
        self.use_self_play = use_self_play
        
        # Inicializar agente V6
        if AGENT_AVAILABLE:
            config = AgentConfig(
                grid_size=30,  # Tamanho padrão ARC-AGI-3
                num_colors=10,
                max_steps=1000
            )
            self.agent = ARCGeneticBabyV6(config)
            if self.verbose:
                print(f"🧠 ARC Genetic Baby V6 inicializado")
                print(f"   Curiosity: {use_curiosity}")
                print(f"   Developmental: {use_developmental}")
                print(f"   Self-Play: {use_self_play}")
        else:
            self.agent = None
            print("❌ Agente V6 não disponível")
        
        # Mapeamento de ações
        self.action_map = {
            'up': GameAction.ACTION0,
            'down': GameAction.ACTION1,
            'left': GameAction.ACTION2,
            'right': GameAction.ACTION3,
            'select': GameAction.ACTION4,
            'submit': GameAction.ACTION5,
            'clear': GameAction.ACTION6,
            'undo': GameAction.ACTION7,
        }
        
        # Estatísticas
        self.episode_count = 0
        self.total_steps = 0
        self.successful_episodes = 0
        
    def step(self, frame_data) -> int:
        """
        Passo principal - recebe FrameData e retorna GameAction
        
        Args:
            frame_data: Estado atual do jogo (FrameData)
            
        Returns:
            int: Ação a executar (GameAction)
        """
        self.total_steps += 1
        
        # Converter FrameData para formato do V6
        grid_state = self._convert_frame_to_grid(frame_data)
        
        # Processar com agente V6
        if self.agent:
            try:
                # Usar sistema de decisão do V6
                action_name, reasoning = self.agent.decide_action(grid_state)
                
                # Mapear ação do V6 para GameAction
                if action_name in self.action_map:
                    return self.action_map[action_name]
                else:
                    # Fallback: ação aleatória das disponíveis
                    import random
                    available = getattr(frame_data, 'available_actions', [0,1,2,3,4,5,6,7])
                    return random.choice(available)
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Erro no agente V6: {e}")
                # Fallback para ação básica
                return self._fallback_action(frame_data)
        else:
            return self._fallback_action(frame_data)
    
    def _convert_frame_to_grid(self, frame_data) -> np.ndarray:
        """Converte FrameData para grid numpy"""
        grid = np.array(frame_data.grid)
        
        # Garantir formato correto
        if len(grid.shape) == 1:
            # Grid linear, converter para 2D
            size = int(np.sqrt(len(grid)))
            grid = grid.reshape(size, size)
        
        return grid
    
    def _fallback_action(self, frame_data) -> int:
        """Ação de fallback quando agente falha"""
        # Ação simples: tentar completar nível
        available = getattr(frame_data, 'available_actions', [5])  # ACTION5 = submit
        
        # Se temos níveis completados vs objetivo, tentar submit
        completed = getattr(frame_data, 'levels_completed', 0)
        target = getattr(frame_data, 'win_levels', 1)
        
        if completed >= target and 5 in available:
            return 5  # Submit
        
        # Caso contrário, explorar
        import random
        return random.choice(available)
    
    def on_episode_start(self, level_id: str = None):
        """Callback no início de episódio"""
        self.episode_count += 1
        if self.verbose:
            print(f"🎮 Episódio {self.episode_count} iniciado" + 
                  (f" (Level: {level_id})" if level_id else ""))
        
        # Resetar estado do agente se necessário
        if self.agent and hasattr(self.agent, 'reset_episode'):
            self.agent.reset_episode()
    
    def on_episode_end(self, success: bool, score: float = 0.0):
        """Callback no fim de episódio"""
        if success:
            self.successful_episodes += 1
        
        if self.verbose:
            status = "✅ Sucesso" if success else "❌ Falha"
            print(f"🏁 Episódio finalizado: {status} (Score: {score:.2f})")
            
        # Aprender com resultado
        if self.agent and hasattr(self.agent, 'learn'):
            self.agent.learn(success=success, reward=score)
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do agente"""
        stats = {
            'episodes': self.episode_count,
            'successful': self.successful_episodes,
            'success_rate': self.successful_episodes / max(1, self.episode_count),
            'total_steps': self.total_steps,
            'sdk_available': SDK_AVAILABLE,
            'agent_available': AGENT_AVAILABLE,
        }
        
        if self.agent and hasattr(self.agent, 'get_stats'):
            stats['v6_stats'] = self.agent.get_stats()
        
        return stats


# Compatibilidade com sistema de execução da competição
def create_agent(**kwargs) -> ARCGeneticBabyV6Adapter:
    """Factory function para criar agente"""
    return ARCGeneticBabyV6Adapter(**kwargs)


# Teste local (se executado diretamente)
if __name__ == "__main__":
    print("🧪 Testando ARC Genetic Baby V6 Adapter")
    print(f"SDK ARC-AGI disponível: {SDK_AVAILABLE}")
    print(f"Agente V6 disponível: {AGENT_AVAILABLE}")
    
    # Criar agente
    agent = create_agent(verbose=True)
    
    # Testar com dados simulados
    test_frame = FrameData(
        grid=[[0]*30 for _ in range(30)],
        available_actions=[0,1,2,3,4,5,6,7],
        levels_completed=0,
        win_levels=1
    )
    
    print("\n🎮 Simulando passo...")
    action = agent.step(test_frame)
    print(f"Ação escolhida: {action}")
    
    print("\n📊 Estatísticas:")
    print(agent.get_stats())
