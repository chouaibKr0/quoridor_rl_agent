"""
Pygame UI Renderer Module for Quoridor.
Provides full visual rendering for board, pawns, walls, valid moves, and menu interaction.
"""

import pygame
import numpy as np


class QuoridorUI:
    def __init__(self, game):
        pygame.init()
        pygame.font.init()
        self.game = game
        
        # Display settings
        self.cell_size = 60
        self.margin = 100
        self.board_size = self.cell_size * 9
        self.info_width = 300
        
        self.width = self.margin * 2 + self.board_size + self.info_width
        self.height = self.margin * 2 + self.board_size
        
        # Create window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Quoridor V2")
        
        # Colors
        self.BG_COLOR = (40, 44, 52)
        self.BOARD_COLOR = (60, 63, 70)
        self.GRID_COLOR = (80, 86, 95)
        self.P1_COLOR = (52, 152, 219)  # Blue
        self.P2_COLOR = (231, 76, 60)   # Red
        self.WALL_COLOR = (149, 165, 166)
        self.HIGHLIGHT_COLOR = (46, 204, 113)
        self.TEXT_COLOR = (236, 240, 241)
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 28)
        self.tiny_font = pygame.font.Font(None, 20)
        
        # UI state
        self.selected_cell = None
        self.mode = "move"  # "move" or "wall"
        self.wall_orientation = "h"  # "h" or "v"
        self.hovered_wall = None
        
        # Menu state
        self.start_btn_rect = pygame.Rect(self.width // 2 - 100, self.height - 150, 200, 60)
        self.p1_rect = pygame.Rect(self.margin, 200, 300, 50)
        self.p2_rect = pygame.Rect(self.width - self.margin - 300, 200, 300, 50)
        
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.game.reset()
                self.mode = "move"
            elif event.key == pygame.K_m:
                self.mode = "move"
            elif event.key == pygame.K_w:
                self.mode = "wall"
            elif event.key == pygame.K_h and self.mode == "wall":
                self.wall_orientation = "h"
            elif event.key == pygame.K_v and self.mode == "wall":
                self.wall_orientation = "v"
                
        elif event.type == pygame.MOUSEBUTTONDOWN and not self.game.done:
            if self.mode == "move":
                self.handle_move_click(event.pos)
            elif self.mode == "wall":
                self.handle_wall_click(event.pos)
                
        elif event.type == pygame.MOUSEMOTION:
            if self.mode == "wall":
                self.update_wall_hover(event.pos)
    
    def handle_move_click(self, pos):
        mx, my = pos
        if mx < self.margin or mx > self.margin + self.board_size:
            return
        if my < self.margin or my > self.margin + self.board_size:
            return
        
        col = (mx - self.margin) // self.cell_size
        row = (my - self.margin) // self.cell_size
        
        if row < 0 or row >= 9 or col < 0 or col >= 9:
            return
        
        target = (row, col)
        curr = self.game.p1_pos if self.game.current_player == 1 else self.game.p2_pos
        
        dr = target[0] - curr[0]
        dc = target[1] - curr[1]
        
        action_map = {
            (-1, 0): 0,   # North
            (0, 1): 1,    # East
            (1, 0): 2,    # South
            (0, -1): 3,   # West
            (-2, 0): 4,   # Jump North
            (0, 2): 5,    # Jump East
            (2, 0): 6,    # Jump South
            (0, -2): 7,   # Jump West
            (-1, 1): 8,   # Slide NE
            (-1, -1): 9,  # Slide NW
            (1, 1): 10,   # Slide SE
            (1, -1): 11,  # Slide SW
        }
        
        if (dr, dc) in action_map:
            action = action_map[(dr, dc)]
            legal_moves = self.game.get_legal_moves()
            if legal_moves[action] == 1.0:
                try:
                    self.game.step(action)
                except Exception:
                    pass
    
    def handle_wall_click(self, pos):
        if self.hovered_wall is None:
            return
        
        wall_type, r, c = self.hovered_wall
        
        if wall_type == 'h':
            idx = r * 8 + c
            action = 12 + idx
        else:
            idx = r * 8 + c
            action = 12 + 64 + idx
        
        legal_moves = self.game.get_legal_moves()
        if action < 140 and legal_moves[action] == 1.0:
            try:
                self.game.step(action)
            except Exception:
                pass
    
    def update_wall_hover(self, pos):
        mx, my = pos
        rel_x = mx - self.margin
        rel_y = my - self.margin
        
        if rel_x < 0 or rel_x >= self.board_size or rel_y < 0 or rel_y >= self.board_size:
            self.hovered_wall = None
            return
        
        if self.wall_orientation == 'h':
            row = round(rel_y / self.cell_size) - 1
            col = int(rel_x / self.cell_size)
            row = max(0, min(7, row))
            col = max(0, min(7, col))
            self.hovered_wall = ('h', row, col)
        else:
            row = int(rel_y / self.cell_size)
            col = round(rel_x / self.cell_size) - 1
            row = max(0, min(7, row))
            col = max(0, min(7, col))
            self.hovered_wall = ('v', row, col)
    
    def draw(self):
        self.screen.fill(self.BG_COLOR)
        
        self.draw_board()
        self.draw_walls()
        
        if self.mode == "wall" and self.hovered_wall:
            self.draw_wall_preview()
        
        if self.mode == "move":
            self.draw_valid_moves()
        
        self.draw_players()
        self.draw_info_panel()
        
        pygame.display.flip()
    
    def draw_board(self):
        board_rect = pygame.Rect(self.margin, self.margin, self.board_size, self.board_size)
        pygame.draw.rect(self.screen, self.BOARD_COLOR, board_rect)
        
        for i in range(10):
            x = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, self.margin), (x, self.margin + self.board_size), 2)
            y = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.GRID_COLOR, (self.margin, y), (self.margin + self.board_size, y), 2)
    
    def draw_walls(self):
        for r, c in self.game.walls_h:
            x = self.margin + c * self.cell_size
            y = self.margin + (r + 1) * self.cell_size
            pygame.draw.rect(self.screen, self.WALL_COLOR, (x, y - 6, self.cell_size * 2, 12))
            pygame.draw.rect(self.screen, (200, 200, 200), (x, y - 6, self.cell_size * 2, 12), 1)
        
        for r, c in self.game.walls_v:
            x = self.margin + (c + 1) * self.cell_size
            y = self.margin + r * self.cell_size
            pygame.draw.rect(self.screen, self.WALL_COLOR, (x - 6, y, 12, self.cell_size * 2))
            pygame.draw.rect(self.screen, (200, 200, 200), (x - 6, y, 12, self.cell_size * 2), 1)
    
    def draw_wall_preview(self):
        wall_type, r, c = self.hovered_wall
        if wall_type == 'h':
            idx = r * 8 + c
            action = 12 + idx
        else:
            idx = r * 8 + c
            action = 12 + 64 + idx
        
        legal_moves = self.game.get_legal_moves()
        is_legal = action < 140 and legal_moves[action] == 1.0
        color = self.HIGHLIGHT_COLOR if is_legal else (231, 76, 60)
        
        if wall_type == 'h':
            x = self.margin + c * self.cell_size
            y = self.margin + (r + 1) * self.cell_size
            pygame.draw.rect(self.screen, color, (x, y - 6, self.cell_size * 2, 12))
            pygame.draw.rect(self.screen, (255, 255, 255), (x, y - 6, self.cell_size * 2, 12), 2)
        else:
            x = self.margin + (c + 1) * self.cell_size
            y = self.margin + r * self.cell_size
            pygame.draw.rect(self.screen, color, (x - 6, y, 12, self.cell_size * 2))
            pygame.draw.rect(self.screen, (255, 255, 255), (x - 6, y, 12, self.cell_size * 2), 2)
    
    def draw_valid_moves(self):
        curr = self.game.p1_pos if self.game.current_player == 1 else self.game.p2_pos
        opp = self.game.p2_pos if self.game.current_player == 1 else self.game.p1_pos
        
        valid_targets = self.game._get_valid_pawn_moves(curr, opp)
        
        for r, c in valid_targets:
            x = self.margin + c * self.cell_size + self.cell_size // 2
            y = self.margin + r * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.HIGHLIGHT_COLOR, (x, y), 8)
    
    def draw_players(self):
        r1, c1 = self.game.p1_pos
        x1 = self.margin + c1 * self.cell_size + self.cell_size // 2
        y1 = self.margin + r1 * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, self.P1_COLOR, (x1, y1), 20)
        pygame.draw.circle(self.screen, (255, 255, 255), (x1, y1), 20, 3)
        
        text = self.tiny_font.render("1", True, (255, 255, 255))
        text_rect = text.get_rect(center=(x1, y1))
        self.screen.blit(text, text_rect)
        
        r2, c2 = self.game.p2_pos
        x2 = self.margin + c2 * self.cell_size + self.cell_size // 2
        y2 = self.margin + r2 * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, self.P2_COLOR, (x2, y2), 20)
        pygame.draw.circle(self.screen, (255, 255, 255), (x2, y2), 20, 3)
        
        text = self.tiny_font.render("2", True, (255, 255, 255))
        text_rect = text.get_rect(center=(x2, y2))
        self.screen.blit(text, text_rect)
    
    def draw_info_panel(self):
        panel_x = self.margin * 2 + self.board_size + 20
        y_offset = self.margin
        
        title = self.font.render("QUORIDOR V2", True, self.TEXT_COLOR)
        self.screen.blit(title, (panel_x, y_offset))
        y_offset += 60
        
        player_color = self.P1_COLOR if self.game.current_player == 1 else self.P2_COLOR
        player_text = f"Player {self.game.current_player}"
        text = self.small_font.render(player_text, True, player_color)
        self.screen.blit(text, (panel_x, y_offset))
        y_offset += 50
        
        p1_walls_text = f"P1 Walls: {self.game.p1_walls_left}"
        text = self.small_font.render(p1_walls_text, True, self.P1_COLOR)
        self.screen.blit(text, (panel_x, y_offset))
        y_offset += 35
        
        p2_walls_text = f"P2 Walls: {self.game.p2_walls_left}"
        text = self.small_font.render(p2_walls_text, True, self.P2_COLOR)
        self.screen.blit(text, (panel_x, y_offset))
        y_offset += 60
        
        mode_text = f"Mode: {self.mode.upper()}"
        mode_color = self.HIGHLIGHT_COLOR if self.mode == "wall" else self.TEXT_COLOR
        text = self.small_font.render(mode_text, True, mode_color)
        self.screen.blit(text, (panel_x, y_offset))
        y_offset += 35
        
        if self.mode == "wall":
            orient_text = f"Orient: {self.wall_orientation.upper()}"
            orient_color = self.HIGHLIGHT_COLOR
            text = self.small_font.render(orient_text, True, orient_color)
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 50
        else:
            y_offset += 50
        
        y_offset += 20
        controls = [
            "Controls:",
            "M - Move mode",
            "W - Wall mode",
            "H - Horizontal",
            "V - Vertical",
            "R - Reset",
            "ESC - Menu",
        ]
        for control in controls:
            text = self.tiny_font.render(control, True, self.TEXT_COLOR)
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 25

    def draw_menu(self, p1_desc, p2_desc):
        self.screen.fill(self.BG_COLOR)
        
        title = self.font.render("QUORIDOR V2", True, self.HIGHLIGHT_COLOR)
        title_rect = title.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title, title_rect)
        
        p1_title = self.small_font.render("Player 1", True, self.P1_COLOR)
        self.screen.blit(p1_title, (self.margin, 160))
        
        pygame.draw.rect(self.screen, self.BOARD_COLOR, self.p1_rect)
        pygame.draw.rect(self.screen, self.P1_COLOR, self.p1_rect, 2)
        
        if len(p1_desc) > 23:
            p1_desc = p1_desc[:20] + "..."
        text = self.small_font.render(p1_desc, True, self.TEXT_COLOR)
        text_rect = text.get_rect(center=self.p1_rect.center)
        self.screen.blit(text, text_rect)
        
        p2_title = self.small_font.render("Player 2", True, self.P2_COLOR)
        p2_title_rect = p2_title.get_rect(topleft=(self.width - self.margin - 300, 160))
        self.screen.blit(p2_title, p2_title_rect)
        
        pygame.draw.rect(self.screen, self.BOARD_COLOR, self.p2_rect)
        pygame.draw.rect(self.screen, self.P2_COLOR, self.p2_rect, 2)
        
        if len(p2_desc) > 23:
            p2_desc = p2_desc[:20] + "..."
        text = self.small_font.render(p2_desc, True, self.TEXT_COLOR)
        text_rect = text.get_rect(center=self.p2_rect.center)
        self.screen.blit(text, text_rect)
        
        instr = self.tiny_font.render("Click box to cycle solver types. Right click for previous.", True, (150, 150, 150))
        instr_rect = instr.get_rect(center=(self.width // 2, 280))
        self.screen.blit(instr, instr_rect)
        
        color = self.HIGHLIGHT_COLOR
        pygame.draw.rect(self.screen, color, self.start_btn_rect, border_radius=10)
        
        btn_text = self.font.render("START GAME", True, self.BG_COLOR)
        btn_rect = btn_text.get_rect(center=self.start_btn_rect.center)
        self.screen.blit(btn_text, btn_rect)
        
        pygame.display.flip()

    def draw_game_over(self):
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        if self.game.p1_pos[0] == 8:
            msg = "PLAYER 1 WINS!"
            color = self.P1_COLOR
        elif self.game.p2_pos[0] == 0:
            msg = "PLAYER 2 WINS!"
            color = self.P2_COLOR
        else:
            msg = "GAME OVER"
            color = self.TEXT_COLOR
        
        text = self.font.render(msg, True, color)
        rect = text.get_rect(center=(self.width // 2, self.height // 2 - 40))
        self.screen.blit(text, rect)
        
        sub = self.small_font.render("Press R to Restart or ESC for Menu", True, (255, 255, 255))
        sub_rect = sub.get_rect(center=(self.width // 2, self.height // 2 + 20))
        self.screen.blit(sub, sub_rect)
        
        pygame.display.flip()
