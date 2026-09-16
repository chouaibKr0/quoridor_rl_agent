"""
Backward compatibility shim for AI Agents.
Re-exports BaseAgent, heuristic agents, search agents, RL agents, and factory functions from the solvers package.
"""

from solvers.base import BaseAgent
from solvers.heuristic.random_agent import RandomAgent
from solvers.heuristic.dijkstra_agent import DijkstraAgent
from solvers.heuristic.strategic_agent import StrategicAgent
from solvers.search.minimax_agent import MinimaxAgent
from solvers.search.alphabeta_agent import AlphaBetaAgent
from solvers.search.mcts_agent import MCTSAgent
from solvers.rl.ppo_agent import PPOAgent, RLAgent
from solvers.registry import get_solver, get_agent

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "DijkstraAgent",
    "StrategicAgent",
    "MinimaxAgent",
    "AlphaBetaAgent",
    "MCTSAgent",
    "PPOAgent",
    "RLAgent",
    "get_agent",
    "get_solver",
]