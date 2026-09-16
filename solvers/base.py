"""
Base Agent Interface.
Defines the abstract base class for all Quoridor solvers.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all Quoridor agents/solvers."""

    @abstractmethod
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select an action given the current observation and legal action mask.
        
        Args:
            observation: The 9x9x6 game state tensor
            action_mask: Binary mask of shape (140,) where 1 = valid action
            
        Returns:
            Action index (0-139)
        """
        pass

    def reset(self):
        """Reset agent internal state between episodes."""
        pass

    def get_config(self) -> dict:
        """
        Return solver hyperparameters as a flat dict (for logging & introspection).
        Subclasses should override this method to report their configuration.
        """
        return {}
