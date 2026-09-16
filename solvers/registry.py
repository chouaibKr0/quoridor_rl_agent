"""
Solver Registry Module.
Provides centralized registration and instantiation for all Quoridor solvers.
"""

from typing import Dict, Type, Any, Optional
from solvers.base import BaseAgent

SOLVER_REGISTRY: Dict[str, Type[BaseAgent]] = {}


def register_solver(name: str, cls: Type[BaseAgent]):
    """
    Register a solver class under a unique name key.
    
    Args:
        name: String identifier (e.g. 'random', 'dijkstra', 'strategic')
        cls: BaseAgent subclass
    """
    key = name.lower()
    SOLVER_REGISTRY[key] = cls


def get_solver(name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseAgent:
    """
    Instantiate a solver by name with configuration parameters.
    
    Args:
        name: Name key of the registered solver
        config: Dictionary of hyperparameter configurations
        **kwargs: Direct keyword arguments override config values
        
    Returns:
        An instance of the requested BaseAgent subclass
    """
    # Ensure heuristic solvers are imported and registered
    _ensure_solvers_registered()

    key = name.lower()
    if key not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver: '{name}'. Available solvers: {list(SOLVER_REGISTRY.keys())}"
        )

    cls = SOLVER_REGISTRY[key]
    merged_kwargs = {}

    if config is not None:
        merged_kwargs.update(config)
    merged_kwargs.update(kwargs)

    # Validate against CONFIG_SCHEMA if class defines it
    if hasattr(cls, "CONFIG_SCHEMA") and isinstance(cls.CONFIG_SCHEMA, dict):
        schema = cls.CONFIG_SCHEMA
        final_kwargs = {}
        for param, default_val in schema.items():
            if param in merged_kwargs:
                final_kwargs[param] = merged_kwargs[param]
            elif default_val is not None:
                final_kwargs[param] = default_val
        
        return cls(**final_kwargs)

    return cls(**merged_kwargs)


def get_agent(name: str, player: int = 2, seed: Optional[int] = None, model_path: Optional[str] = None) -> BaseAgent:
    """
    Backward-compatibility factory function matching V1 `ai_agent.get_agent` signature.
    """
    _ensure_solvers_registered()
    key = name.lower()

    if key == "rl":
        # Deferred import / call to ai_agent or solvers.rl
        from ai_agent import RLAgent
        return RLAgent(model_path=model_path, player=player, seed=seed)
    elif key == "minimax":
        from ai_agent import MinimaxAgent
        return MinimaxAgent(player=player, seed=seed)
    elif key == "mixed":
        import numpy as np
        from solvers.heuristic.dijkstra_agent import DijkstraAgent
        from solvers.heuristic.strategic_agent import StrategicAgent
        rng = np.random.default_rng(seed)
        if rng.random() < 0.6:
            return DijkstraAgent(player=player, seed=seed)
        else:
            return StrategicAgent(player=player, seed=seed)

    # Standard registry lookup
    cfg = {}
    if player is not None:
        cfg["player"] = player
    if seed is not None:
        cfg["seed"] = seed
    return get_solver(key, config=cfg)


def _ensure_solvers_registered():
    """Import subpackages to trigger module-level register_solver calls."""
    try:
        import solvers.heuristic.random_agent
        import solvers.heuristic.dijkstra_agent
        import solvers.heuristic.strategic_agent
    except ImportError:
        pass
