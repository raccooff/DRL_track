"""
Play Tetris with falling pieces

Usage:
    Human play:   python play.py --mode manual
    AI play:      python play.py --mode agent --model_path tetris_dqn.pt

Human controls:
    Left / Right  — move piece
    Up            — rotate
    Down          — soft drop (speed up fall)
    Space         — hard drop (instant)
    Q             — quit
"""

import argparse
import time

import numpy as np
import pygame

from environment import TetrisEnv, TETROMINOES, COLORS

from model import DQN
import torch

# color palette
GB_LIGHTEST = (155, 188, 15)   # #9BBC0F  — background / empty cells
GB_LIGHT    = (139, 172, 15)   # #8BAC0F  — grid lines, ghost piece
GB_DARK     = (48,  98,  48)   # #306230  — active piece, text
GB_DARKEST  = (15,  56,  15)   # #0F380F  — locked blocks, borders

# pygame layout constants
CELL_SIZE = 30
SIDEBAR_WIDTH = 200
GHOST_ALPHA = 100

# gravity: parameter for manual playing
GRAVITY_DELAY_MANUAL = 450   # manual time fall
GRAVITY_DELAY_AGENT = 100      # agent time fall

# border and padding
BORDER_WIDTH = 3
OUTER_PADDING = 6

def draw_board(screen, env, offset_x=0, offset_y=0):

    """
        draw board 
    """

    # rectangular coordinates
    board_rect = pygame.Rect(
        offset_x - BORDER_WIDTH,
        offset_y - BORDER_WIDTH,
        env.cols * CELL_SIZE + BORDER_WIDTH * 2,
        env.rows * CELL_SIZE + BORDER_WIDTH * 2,
    )

    # draw background and border
    pygame.draw.rect(screen, GB_DARKEST, board_rect, BORDER_WIDTH)

    for r in range(env.rows):
        for c in range(env.cols):

            rect = pygame.Rect(
                offset_x + c * CELL_SIZE,
                offset_y + r * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )

            if env.board[r][c]:

                # locked block: darkest fill with a light inner bevel
                pygame.draw.rect(screen, GB_DARKEST, rect)
                inner = rect.inflate(-4, -4)
                pygame.draw.rect(screen, GB_DARK, inner)

                # highlight on top-left edges for a raised look
                pygame.draw.line(screen, GB_LIGHT, rect.topleft, rect.topright, 1)
                pygame.draw.line(screen, GB_LIGHT, rect.topleft, rect.bottomleft, 1)

            else:

                # empty cell: lightest fill
                pygame.draw.rect(screen, GB_LIGHTEST, rect)
                pygame.draw.rect(screen, GB_LIGHT, rect, 1)


def draw_active_piece(screen, env, offset_x=0, offset_y=0):

    """
        draw the currently falling piece and its ghost.
    """

    if not env.piece_active:
        return
    
    shape = env._current_shape()

    # ghost piece (landing preview)

    ghost_row = env.get_ghost_row()

    for dr, dc in shape:

        r, c = ghost_row + dr, env.piece_col + dc

        if 0 <= r < env.rows and 0 <= c < env.cols:

            rect = pygame.Rect(
                offset_x + c * CELL_SIZE,
                offset_y + r * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )

            # subtle fill between lightest and light
            ghost_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            ghost_surface.fill((*GB_DARK, GHOST_ALPHA))

            screen.blit(ghost_surface, rect.topleft)

            pygame.draw.rect(screen, GB_DARK, rect, 1)

    # Active falling piece — dark fill with bevel
    for dr, dc in shape:

        r, c = env.piece_row + dr, env.piece_col + dc

        if 0 <= r < env.rows and 0 <= c < env.cols:

            rect = pygame.Rect(
                offset_x + c * CELL_SIZE,
                offset_y + r * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )

            pygame.draw.rect(screen, GB_DARK, rect)

            inner = rect.inflate(-4, -4)
            pygame.draw.rect(screen, GB_DARKEST, inner)

            pygame.draw.line(screen, GB_LIGHT, rect.topleft, rect.topright, 1)
            pygame.draw.line(screen, GB_LIGHT, rect.topleft, rect.bottomleft, 1)


def draw_piece_preview(screen, piece_name, x, y, label, font):

    """
        draw a small piece preview in the sidebar
    """

    text = font.render(label, True, GB_DARKEST)
    screen.blit(text, (x, y))
    shape = TETROMINOES[piece_name][0]
    preview_cell = 18

    # draw a small background box
    min_r = min(dr for dr, dc in shape)
    max_r = max(dr for dr, dc in shape)
    min_c = min(dc for dr, dc in shape)
    max_c = max(dc for dr, dc in shape)

    box_w = (max_c - min_c + 1) * preview_cell + 8
    box_h = (max_r - min_r + 1) * preview_cell + 8

    box_rect = pygame.Rect(x - 2, y + 22, box_w, box_h)

    pygame.draw.rect(screen, GB_LIGHTEST, box_rect)
    pygame.draw.rect(screen, GB_DARKEST, box_rect, 2)

    for dr, dc in shape:

        rect = pygame.Rect(
            x + (dc - min_c) * preview_cell + 2,
            y + 26 + (dr - min_r) * preview_cell,
            preview_cell,
            preview_cell,
        )

        pygame.draw.rect(screen, GB_DARK, rect)
        inner = rect.inflate(-3, -3)
        pygame.draw.rect(screen, GB_DARKEST, inner)
        pygame.draw.rect(screen, GB_LIGHT, rect, 1)


def draw_sidebar(screen, env, font, board_width):

    """
        draw the sidebar with score, lines, and piece previews
    """

    x = board_width + OUTER_PADDING + 20

    # Title
    title_font = pygame.font.SysFont("monospace", 22, bold=True)
    title = title_font.render("TETRIS", True, GB_DARKEST)
    screen.blit(title, (x, 10))

    # Divider line
    pygame.draw.line(screen, GB_DARKEST, (x, 35), (x + SIDEBAR_WIDTH - 40, 35), 2)

    # Score
    score_label = font.render("SCORE", True, GB_DARKEST)
    score_value = font.render(f"{env.score:>8}", True, GB_DARK)
    screen.blit(score_label, (x, 45))
    screen.blit(score_value, (x, 65))

    # Lines
    lines_label = font.render("LINES", True, GB_DARKEST)
    lines_value = font.render(f"{env.lines_cleared:>8}", True, GB_DARK)
    screen.blit(lines_label, (x, 95))
    screen.blit(lines_value, (x, 115))

    # Divider
    pygame.draw.line(screen, GB_DARKEST, (x, 145), (x + SIDEBAR_WIDTH - 40, 145), 2)

    # Current piece
    draw_piece_preview(screen, env.current_piece, x, 155, "CURRENT", font)

    # Next piece
    draw_piece_preview(screen, env.next_piece, x, 250, "NEXT", font)


def draw_game_over(screen, board_width, board_height, offset_x=0, offset_y=0):
    """Draw the Game Over overlay in Game Boy style."""
    # Semi-transparent overlay
    overlay = pygame.Surface((board_width, board_height), pygame.SRCALPHA)
    overlay.fill((*GB_LIGHTEST, 180))
    screen.blit(overlay, (offset_x, offset_y))

    # Box
    box_w, box_h = 220, 80
    box_x = offset_x + (board_width - box_w) // 2
    box_y = offset_y + (board_height - box_h) // 2
    box_rect = pygame.Rect(box_x, box_y, box_w, box_h)
    pygame.draw.rect(screen, GB_LIGHTEST, box_rect)
    pygame.draw.rect(screen, GB_DARKEST, box_rect, 3)

    font_big = pygame.font.SysFont("monospace", 30, bold=True)
    text = font_big.render("GAME OVER", True, GB_DARKEST)
    text_rect = text.get_rect(center=box_rect.center)
    screen.blit(text, text_rect)


def draw_background(screen, total_width, total_height):
    """Fill the entire window with the Game Boy light background."""
    screen.fill(GB_LIGHT)
    # Draw a subtle outer border
    pygame.draw.rect(
        screen, GB_DARKEST,
        pygame.Rect(0, 0, total_width, total_height),
        BORDER_WIDTH,
    )


# ── Manual play ───────────────────────────────────────────────────────

def play_manual(env):

    board_width = env.cols * CELL_SIZE
    board_height = env.rows * CELL_SIZE

    total_width = board_width + SIDEBAR_WIDTH + OUTER_PADDING * 2
    total_height = board_height + OUTER_PADDING * 2
    screen = pygame.display.set_mode((total_width, total_height))

    pygame.display.set_caption("TETRIS")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)

    env.reset()
    env.spawn_piece()

    GRAVITY_EVENT = pygame.USEREVENT + 1

    pygame.time.set_timer(GRAVITY_EVENT, GRAVITY_DELAY_MANUAL)
    pygame.key.set_repeat(170, 50)

    running = True

    while running and not env.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == GRAVITY_EVENT:
                env.tick()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    env.move_left()
                elif event.key == pygame.K_RIGHT:
                    env.move_right()
                elif event.key == pygame.K_UP:
                    env.rotate()
                elif event.key == pygame.K_DOWN:
                    env.soft_drop()
                elif event.key == pygame.K_SPACE:
                    env.hard_drop_live()
                elif event.key == pygame.K_q:
                    running = False

        # ── Draw ──
        draw_background(screen, total_width, total_height)
        draw_board(screen, env, OUTER_PADDING, OUTER_PADDING)
        draw_active_piece(screen, env, OUTER_PADDING, OUTER_PADDING)
        draw_sidebar(screen, env, font, board_width + OUTER_PADDING)

        if env.game_over:
            draw_game_over(screen, board_width, board_height, OUTER_PADDING, OUTER_PADDING)

        pygame.display.flip()
        clock.tick(60)

    # Hold on game-over screen
    if env.game_over:
        draw_background(screen, total_width, total_height)
        draw_board(screen, env, OUTER_PADDING, OUTER_PADDING)
        draw_sidebar(screen, env, font, board_width + OUTER_PADDING)
        draw_game_over(screen, board_width, board_height, OUTER_PADDING, OUTER_PADDING)
        pygame.display.flip()
        time.sleep(3)

# ── Agent play ───────────────────────────────────────────────────────

def play_agent(env, model_path):

    board_width = env.cols * CELL_SIZE
    board_height = env.rows * CELL_SIZE

    total_width = board_width + SIDEBAR_WIDTH + OUTER_PADDING * 2
    total_height = board_height + OUTER_PADDING * 2

    screen = pygame.display.set_mode((total_width, total_height))

    pygame.display.set_caption("TETRIS - AGENT")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN(env.state_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env.reset()
    env.spawn_piece()

    # agent timer
    AGENT_EVENT = pygame.USEREVENT + 2
    pygame.time.set_timer(AGENT_EVENT, GRAVITY_DELAY_AGENT)

    running = True

    while running and not env.game_over:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:

                running = False

            elif event.type == AGENT_EVENT:

                actions = env.get_possible_actions()

                if not actions:

                    env.game_over = True
                    break

                states = []
                valid_actions = []
                for a in actions:

                    s = env.get_state_for_action(a)

                    if s is not None:

                        states.append(s)
                        valid_actions.append(a)

                if not valid_actions:

                    env.game_over = True
                    break

                with torch.no_grad():

                    x = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                    q = model(x).squeeze(1)

                    best_idx = torch.argmax(q).item()

                best_action = valid_actions[best_idx]

                _, _, done, _ = env.step(best_action)

                if done:
                    env.game_over = True

        draw_background(screen, total_width, total_height)
        draw_board(screen, env, OUTER_PADDING, OUTER_PADDING)

        # no active falling piece
        draw_sidebar(screen, env, font, board_width + OUTER_PADDING)

        if env.game_over:
            
            draw_game_over(screen, board_width, board_height, OUTER_PADDING, OUTER_PADDING)

        pygame.display.flip()
        clock.tick(60)

    if env.game_over:
        draw_background(screen, total_width, total_height)
        draw_board(screen, env, OUTER_PADDING, OUTER_PADDING)
        draw_sidebar(screen, env, font, board_width + OUTER_PADDING)
        draw_game_over(screen, board_width, board_height, OUTER_PADDING, OUTER_PADDING)
        pygame.display.flip()
        time.sleep(3)

def main():

    parser = argparse.ArgumentParser(description="Play Tetris")

    parser.add_argument(
        "--mode", 
        type=str, 
        default="manual", 
        choices=["manual", "agent"],
        help="'manual' for manual play, 'agent' for testing",
    )

    parser.add_argument("--model_path", type=str, default="", help="path to model")

    args = parser.parse_args()

    pygame.init()
    env = TetrisEnv(rows=20, cols=10)

    try:

        if args.mode == "manual":

            play_manual(env)

        else:

            if not args.model_path:
                
                raise ValueError("Please pass --model_path for agent mode.")
            
            play_agent(env, args.model_path)

    finally:
        
        pygame.quit()


if __name__ == "__main__":

    main()