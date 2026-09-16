"""
Game Replay Module for Quoridor.
Replays saved experiment JSON game records step-by-step in Pygame with telemetry overlay.
"""

import sys
import json
import time
import pygame
from typing import Optional, Dict, Any

from core.game import QuoridorGame
from ui.ui import QuoridorUI


class GameReplayer:
    """
    Step-by-step game replayer for experiment result JSON files.
    """

    def __init__(self, json_file_path: str):
        self.json_path = json_file_path
        with open(json_file_path, "r") as f:
            self.game_data = json.load(f)

        self.moves = self.game_data.get("moves", [])
        self.total_moves = len(self.moves)
        self.current_step = 0

        self.game = QuoridorGame()
        self.ui = QuoridorUI(self.game)

        self.paused = True
        self.play_speed_fps = 2  # moves per second

    def run(self):
        """Main replay loop."""
        pygame.init()
        clock = pygame.time.Clock()
        running = True
        last_step_t = time.time()

        self._update_game_to_step(self.current_step)

        while running:
            dt = clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused

                    elif event.key == pygame.K_RIGHT:
                        if self.current_step < self.total_moves:
                            self.current_step += 1
                            self._update_game_to_step(self.current_step)

                    elif event.key == pygame.K_LEFT:
                        if self.current_step > 0:
                            self.current_step -= 1
                            self._update_game_to_step(self.current_step)

                    elif event.key == pygame.K_UP:
                        self.play_speed_fps = min(20, self.play_speed_fps + 1)

                    elif event.key == pygame.K_DOWN:
                        self.play_speed_fps = max(1, self.play_speed_fps - 1)

                    elif event.key == pygame.K_r:
                        self.current_step = 0
                        self._update_game_to_step(self.current_step)

            # Auto-play step logic
            if not self.paused and self.current_step < self.total_moves:
                if time.time() - last_step_t >= (1.0 / self.play_speed_fps):
                    self.current_step += 1
                    self._update_game_to_step(self.current_step)
                    last_step_t = time.time()

            # Render UI + Replay overlay
            self.ui.draw()
            self._draw_replay_overlay()
            pygame.display.flip()

        pygame.quit()

    def _update_game_to_step(self, step: int):
        """Reset and apply moves up to step index."""
        self.game.reset()
        for i in range(step):
            if i < len(self.moves):
                self.game.step(self.moves[i])

    def _draw_replay_overlay(self):
        """Draw replay info overlay on panel."""
        panel_x = self.ui.margin * 2 + self.ui.board_size + 20
        y = self.ui.margin + 260

        # Draw semi-transparent panel background
        rect = pygame.Rect(panel_x - 5, y - 5, self.ui.info_width - 15, 220)
        pygame.draw.rect(self.ui.screen, (30, 32, 38), rect, border_radius=6)
        pygame.draw.rect(self.ui.screen, self.ui.HIGHLIGHT_COLOR, rect, 1, border_radius=6)

        title = self.ui.small_font.render("REPLAY MODE", True, self.ui.HIGHLIGHT_COLOR)
        self.ui.screen.blit(title, (panel_x, y))
        y += 30

        sa = self.game_data.get("solver_a", "A")
        sb = self.game_data.get("solver_b", "B")
        w = self.game_data.get("winner", "draw")

        t_match = f"Match: {sa} vs {sb}"
        if len(t_match) > 26:
            t_match = t_match[:23] + "..."
        text = self.ui.tiny_font.render(t_match, True, self.ui.TEXT_COLOR)
        self.ui.screen.blit(text, (panel_x, y))
        y += 22

        t_status = f"Status: {'PAUSED' if self.paused else 'PLAYING'} ({self.play_speed_fps}x)"
        text = self.ui.tiny_font.render(t_status, True, (241, 196, 15) if self.paused else (46, 204, 113))
        self.ui.screen.blit(text, (panel_x, y))
        y += 22

        t_step = f"Move: {self.current_step} / {self.total_moves} (Winner: {w.upper()})"
        text = self.ui.tiny_font.render(t_step, True, self.ui.TEXT_COLOR)
        self.ui.screen.blit(text, (panel_x, y))
        y += 30

        # Display current move stats if available
        stats = self._get_stats_for_step(self.current_step)
        if stats:
            el_ms = stats.get("elapsed_ms", 0.0)
            nodes = stats.get("nodes_expanded", None)
            sims = stats.get("simulations_run", None)

            stat_line = f"Time: {el_ms:.1f}ms"
            if nodes is not None:
                stat_line += f" | Nodes: {nodes}"
            if sims is not None:
                stat_line += f" | Sims: {sims}"

            text = self.ui.tiny_font.render(stat_line, True, (200, 200, 200))
            self.ui.screen.blit(text, (panel_x, y))
            y += 22

        y += 10
        controls = ["Controls:", "SPACE: Play/Pause", "LEFT/RIGHT: Step", "UP/DOWN: Speed"]
        for c in controls:
            text = self.ui.tiny_font.render(c, True, (150, 150, 150))
            self.ui.screen.blit(text, (panel_x, y))
            y += 18

    def _get_stats_for_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step <= 0:
            return None
        move_idx = step - 1
        per_move = self.game_data.get("per_move_stats", {})

        # Determine which solver made this move
        is_p1_move = (move_idx % 2 == 0)
        a_is_p1 = self.game_data.get("a_is_player_1", True)

        solver_key = "solver_a" if (is_p1_move == a_is_p1) else "solver_b"
        s_list = per_move.get(solver_key, [])
        solver_move_idx = move_idx // 2

        if solver_move_idx < len(s_list):
            return s_list[solver_move_idx]
        return None


def replay_game_record(file_path: str):
    replayer = GameReplayer(file_path)
    replayer.run()
