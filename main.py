"""
Main Game Controller.
Handles game loop, menu, agent management, and replay mode.
"""

import pygame
import sys
import os
import argparse
import subprocess
import tkinter as tk
from tkinter import filedialog
from typing import Optional, List, Tuple

from core.game import QuoridorGame, flip_observation, flip_action, flip_mask
from solvers.registry import get_agent, get_solver, SOLVER_REGISTRY, _ensure_solvers_registered
from solvers.base import BaseAgent
from ui import QuoridorUI, GameReplayer


class GameController:
    """
    Main game controller that manages the game flow.
    Supports different modes (HvH, HvAI, AIvAI, RL, Replay).
    """
    
    def __init__(self):
        pygame.init()
        _ensure_solvers_registered()
        
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        self.default_model = "quoridor_ppo_final_20260207_114522.zip"
        
        # Base types dynamically built from solver registry
        registry_keys = [k.capitalize() for k in SOLVER_REGISTRY.keys() if k not in ["ppo", "rl"]]
        self.base_types = ["Human"] + sorted(list(set(registry_keys)))
        
        self.rl_options = ["RL: Default", "RL: Select File..."]
        
        self.p1_types = self.base_types + self.rl_options
        self.p2_types = self.base_types + self.rl_options
        
        self.p1_idx = 0  # Default Human
        self.p2_idx = min(2, len(self.p2_types) - 1)  # Default Dijkstra / Second solver
        
        self.state = "menu"  # "menu", "playing", "game_over"
        self.game = None
        self.ui = None
        self.p1_agent: Optional[BaseAgent] = None
        self.p2_agent: Optional[BaseAgent] = None
        
        dummy_game = QuoridorGame()
        self.ui = QuoridorUI(dummy_game)
        
        self.clock = pygame.time.Clock()
        self.running = True

    def run(self):
        """Main application loop."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if self.state == "menu":
                    self.handle_menu_event(event)
                elif self.state == "playing":
                    self.handle_playing_event(event)
                elif self.state == "game_over":
                    self.handle_game_over_event(event)
            
            if self.state == "menu":
                p1_desc = self.p1_types[self.p1_idx]
                p2_desc = self.p2_types[self.p2_idx]
                self.ui.draw_menu(p1_desc, p2_desc)
                
            elif self.state == "playing":
                self.update_game()
                self.ui.draw()
                
            elif self.state == "game_over":
                self.ui.draw_game_over()
            
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

    def handle_menu_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.ui.start_btn_rect.collidepoint(event.pos):
                    self.start_game()
                elif self.ui.p1_rect.collidepoint(event.pos):
                    self.p1_idx = (self.p1_idx + 1) % len(self.p1_types)
                elif self.ui.p2_rect.collidepoint(event.pos):
                    self.p2_idx = (self.p2_idx + 1) % len(self.p2_types)
            elif event.button == 3:
                if self.ui.p1_rect.collidepoint(event.pos):
                    self.p1_idx = (self.p1_idx - 1) % len(self.p1_types)
                elif self.ui.p2_rect.collidepoint(event.pos):
                    self.p2_idx = (self.p2_idx - 1) % len(self.p2_types)

    def handle_playing_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.state = "menu"
                return
            if event.key == pygame.K_r:
                self.reset_game()
                return

        is_p1_human = (self.p1_agent is None)
        is_p2_human = (self.p2_agent is None)
        
        current_player_is_human = False
        if self.game.current_player == 1 and is_p1_human:
            current_player_is_human = True
        elif self.game.current_player == 2 and is_p2_human:
            current_player_is_human = True
            
        if current_player_is_human:
            self.ui.handle_event(event)

    def handle_game_over_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.reset_game()
                self.state = "playing"
            elif event.key == pygame.K_ESCAPE:
                self.state = "menu"

    def start_game(self):
        """Initialize game with selected agents."""
        p1_type = self.p1_types[self.p1_idx]
        p2_type = self.p2_types[self.p2_idx]
        
        p1_model_path = None
        p2_model_path = None
        
        if p1_type == "RL: Select File...":
            path = self._open_file_dialog()
            if not path:
                return
            p1_model_path = path
        elif p1_type == "RL: Default":
            p1_model_path = os.path.join(self.models_dir, self.default_model)
            
        if p2_type == "RL: Select File...":
            path = self._open_file_dialog()
            if not path:
                return
            p2_model_path = path
        elif p2_type == "RL: Default":
            p2_model_path = os.path.join(self.models_dir, self.default_model)

        self.game = QuoridorGame()
        self.ui.game = self.game
        self.ui.mode = "move"
        
        self.p1_agent = self._create_agent(p1_type, 1, p1_model_path)
        self.p2_agent = self._create_agent(p2_type, 2, p2_model_path)
        
        print(f"Starting Game: {p1_type} vs {p2_type}")
        self.state = "playing"

    def _open_file_dialog(self) -> Optional[str]:
        if sys.platform.startswith("linux"):
            try:
                result = subprocess.run(
                    ['zenity', '--file-selection', '--file-filter=*.zip', f'--filename={self.models_dir}/'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode == 0:
                    path = result.stdout.strip()
                    if path:
                        return path
                return None
            except Exception:
                pass

        try:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                initialdir=self.models_dir,
                title="Select RL Model",
                filetypes=[("Zip files", "*.zip"), ("All files", "*.*")]
            )
            root.destroy()
            return file_path if file_path else None
        except Exception as e:
            print(f"Tkinter dialog error: {e}")
            return None

    def reset_game(self):
        self.game.reset()
        if self.p1_agent:
            self.p1_agent.reset()
        if self.p2_agent:
            self.p2_agent.reset()
        self.ui.hovered_wall = None
        self.ui.selected_cell = None
        self.ui.mode = "move"

    def _create_agent(self, type_str: str, player: int, model_path: Optional[str] = None) -> Optional[BaseAgent]:
        if type_str == "Human":
            return None
            
        if type_str.startswith("RL: "):
            return get_agent("rl", player=player, model_path=model_path)
        
        return get_agent(type_str.lower(), player=player)

    def update_game(self):
        if self.game.done:
            self.state = "game_over"
            return

        agent = self.p1_agent if self.game.current_player == 1 else self.p2_agent
            
        if agent is not None:
            obs = self.game._get_observation()
            mask = self.game.get_legal_moves()
            
            if self.game.current_player == 2:
                obs = flip_observation(obs)
                mask = flip_mask(mask)
            
            try:
                action = agent.select_action(obs, mask)
                if self.game.current_player == 2:
                    action = flip_action(action)

                self.game.step(action)
                pygame.time.wait(200)
            except Exception as e:
                print(f"Error AI Agent {agent}: {e}")
                pass


def main():
    parser = argparse.ArgumentParser(description="Quoridor V2 Main Entry Point")
    parser.add_argument("--replay", type=str, default=None, help="Path to experiment game JSON record to replay")

    args = parser.parse_args()

    if args.replay:
        if not os.path.exists(args.replay):
            print(f"Error: Replay file not found: {args.replay}")
            sys.exit(1)
        print(f"Launching Replay Mode for: {args.replay}")
        replayer = GameReplayer(args.replay)
        replayer.run()
    else:
        controller = GameController()
        controller.run()


if __name__ == "__main__":
    main()