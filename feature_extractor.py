"""
Backward compatibility shim for feature extractors.
Re-exports QuoridorCNN and QuoridorResidualCNN from solvers.rl.feature_extractor.
"""

from solvers.rl.feature_extractor import QuoridorCNN, QuoridorResidualCNN

__all__ = ["QuoridorCNN", "QuoridorResidualCNN"]

