"""
Main Game Controller
Handles game loop, menu, and agent management.
"""

import pygame
import sys
import os
import time
from typing import Optional, List, Tuple



import subprocess
import tkinter as tk
from tkinter import filedialog
from quoridor_game import QuoridorGame
from ai_agent import get_agent, BaseAgent, RLAgent
from ui import QuoridorUI

class GameController:
    """
    Main game controller that manages the game flow.
    Supports different modes (HvH, HvAI, AIvAI, RL).
    """
    
    def __init__(self):
        pygame.init()
        
        # models directory
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        # specific default
        self.default_model = "quoridor_ppo_final_20260207_114522.zip"
        
        # Agent types
        # Basic types + any found models
        self.base_types = ["Human", "Random", "Dijkstra", "Strategic", "Minimax"]
        
        # Add special RL options
        self.rl_options = ["RL: Default", "RL: Select File..."]
        
        self.p1_types = self.base_types + self.rl_options
        self.p2_types = self.base_types + self.rl_options
        
        # Selection state
        self.p1_idx = 0  # Default specific Human
        self.p2_idx = 2  # Default Dijkstra
        
        # Game State
        self.state = "menu"  # "menu", "playing", "game_over"
        self.game = None
        self.ui = None
        self.p1_agent: Optional[BaseAgent] = None
        self.p2_agent: Optional[BaseAgent] = None
        
        # Initialize UI (needs a dummy game first for sizing)
        dummy_game = QuoridorGame()
        self.ui = QuoridorUI(dummy_game)
        
        self.clock = pygame.time.Clock()
        self.running = True

    def _scan_models(self) -> List[str]:
        """Scan models directory for .zip files."""
        if not os.path.exists(self.models_dir):
            return []
        files = [f for f in os.listdir(self.models_dir) if f.endswith(".zip")]
        files.sort()
        return files

    def run(self):
        """Main application loop."""
        while self.running:
            # Handle Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if self.state == "menu":
                    self.handle_menu_event(event)
                elif self.state == "playing":
                    self.handle_playing_event(event)
                elif self.state == "game_over":
                    self.handle_game_over_event(event)
            
            # Update & Draw
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
            if event.button == 1: # Left click
                # Check Start Button
                if self.ui.start_btn_rect.collidepoint(event.pos):
                    self.start_game()
                # Check P1 Box
                elif self.ui.p1_rect.collidepoint(event.pos):
                    self.p1_idx = (self.p1_idx + 1) % len(self.p1_types)
                # Check P2 Box
                elif self.ui.p2_rect.collidepoint(event.pos):
                    self.p2_idx = (self.p2_idx + 1) % len(self.p2_types)
            elif event.button == 3: # Right click (previous)
                # Check P1 Box
                if self.ui.p1_rect.collidepoint(event.pos):
                    self.p1_idx = (self.p1_idx - 1) % len(self.p1_types)
                # Check P2 Box
                elif self.ui.p2_rect.collidepoint(event.pos):
                    self.p2_idx = (self.p2_idx - 1) % len(self.p2_types)

    def handle_playing_event(self, event):
        # Allow quitting to menu with 'M' anytime
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                 self.state = "menu"
                 return
            if event.key == pygame.K_r:
                 self.reset_game()
                 return

        # Pass event to UI ONLY if it's Human turn
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
            elif event.key == pygame.K_m:
                self.state = "menu"
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Click anywhere to restart? No, explicitly enforce keys or maybe add buttons later
            pass

    def start_game(self):
        """Initialize game with selected agents."""
        # Check if we need to open file dialog
        p1_type = self.p1_types[self.p1_idx]
        p2_type = self.p2_types[self.p2_idx]
        
        p1_model_path = None
        p2_model_path = None
        
        # Handle P1
        if p1_type == "RL: Select File...":
            path = self._open_file_dialog()
            if not path: return # Cancelled
            p1_model_path = path
        elif p1_type == "RL: Default":
            p1_model_path = os.path.join(self.models_dir, self.default_model)
            
        # Handle P2
        if p2_type == "RL: Select File...":
            path = self._open_file_dialog()
            if not path: return # Cancelled
            p2_model_path = path
        elif p2_type == "RL: Default":
            p2_model_path = os.path.join(self.models_dir, self.default_model)

        self.game = QuoridorGame()
        self.ui.game = self.game # Update UI reference
        self.ui.mode = "move" # Reset UI mode
        
        # Setup Agents
        self.p1_agent = self._create_agent(p1_type, 1, p1_model_path)
        self.p2_agent = self._create_agent(p2_type, 2, p2_model_path)
        
        print(f"Starting Game: {p1_type} vs {p2_type}")
        
        self.state = "playing"

    def _open_file_dialog(self) -> Optional[str]:
        """Open system file dialog to select model."""
        # Check for Linux/Zenity first (native look, no freeze)
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
            except FileNotFoundError:
                # Fallback if zenity not found
                pass
            except Exception as e:
                print(f"Zenity error: {e}")
                pass

        # Fallback to Tkinter (works on Windows/Mac, might freeze on Linux)
        try:
            root = tk.Tk()
            root.withdraw() # Hide main window
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
        """Reset current game."""
        self.game.reset()
        if self.p1_agent: self.p1_agent.reset()
        if self.p2_agent: self.p2_agent.reset()
        # Ensure UI state is reset
        self.ui.hovered_wall = None
        self.ui.selected_cell = None
        self.ui.mode = "move"

    def _create_agent(self, type_str: str, player: int, model_path: Optional[str] = None) -> Optional[BaseAgent]:
        """Factory helper."""
        if type_str == "Human":
            return None
            
        if type_str.startswith("RL: "):
            return get_agent("rl", player=player, model_path=model_path)
        
        # Standard agent
        return get_agent(type_str.lower(), player=player)

    def update_game(self):
        """Game logic update (AI moves)."""
        if self.game.done:
            self.state = "game_over"
            return

        # Check turn
        if self.game.current_player == 1:
            agent = self.p1_agent
        else:
            agent = self.p2_agent
            
        if agent is not None:
            # AI Turn
            # 1. Get observation
            obs = self.game._get_observation()
            # 2. Get action mask
            mask = self.game.get_legal_moves()
            
            # 3. Get action (add small delay for visuals?)
            # check limits to avoid freezing UI
            # For pure AI vs AI visual, maybe limit speed
            # For Human vs AI, instant is OK but jarring
            
            # Simple delay if needed:
            # pygame.time.wait(100) 
            
            try:
                action = agent.select_action(obs, mask)
                # 4. Apply
                self.game.step(action)
                # wait for 0.8s
                pygame.time.wait(800)
            except Exception as e:
                print(f"Error AI Agent {agent}: {e}")
                # Fallback? Pass turn?
                # For now just print error
                pass


def main():
    controller = GameController()
    controller.run()

if __name__ == "__main__":
    main()