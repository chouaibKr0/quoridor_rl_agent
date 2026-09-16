"""
Backward compatibility shim for Numba utilities.
Re-exports functions from core.numba_utils.
"""

from core.numba_utils import (
    get_valid_neighbors_numba,
    bfs_distances_numba,
    check_path_exists_numba,
    get_valid_pawn_moves_numba,
    compute_legal_moves_mask,
)

__all__ = [
    "get_valid_neighbors_numba",
    "bfs_distances_numba",
    "check_path_exists_numba",
    "get_valid_pawn_moves_numba",
    "compute_legal_moves_mask",
]
