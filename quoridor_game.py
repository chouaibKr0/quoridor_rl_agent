"""
Backward compatibility shim for Quoridor Game logic.
Re-exports classes and functions from core.game and core.state.
"""

from core.game import QuoridorGame, flip_observation, flip_action, flip_mask
from core.state import QuoridorStateBuilder
from core.numba_utils import bfs_distances_numba, check_path_exists_numba, get_valid_neighbors_numba

__all__ = [
    "QuoridorGame",
    "QuoridorStateBuilder",
    "flip_observation",
    "flip_action",
    "flip_mask",
    "bfs_distances_numba",
    "check_path_exists_numba",
    "get_valid_neighbors_numba",
]
