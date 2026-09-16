"""
Backward compatibility shim for Quoridor Environment.
Re-exports classes from core.env.
"""

from core.env import QuoridorEnv, SelfPlayEnv

__all__ = [
    "QuoridorEnv",
    "SelfPlayEnv",
]
