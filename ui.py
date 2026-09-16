"""
Backward compatibility shim for Quoridor Pygame UI and Replay.
Re-exports QuoridorUI and GameReplayer from the ui package.
"""

from ui.ui import QuoridorUI
from ui.replay import GameReplayer, replay_game_record

__all__ = ["QuoridorUI", "GameReplayer", "replay_game_record"]