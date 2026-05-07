"""

    Tetris environment.

"""

import numpy as np
import random

# numpy random seed
np.random.seed(0)

# Tetromino shapes
TETROMINOES = {

    'I': [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],

    'O': [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],

    'T': [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (1, -1)],
    ],

    'S': [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],

    'Z': [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],

    'J': [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, -1)],
    ],

    'L': [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],

}

PIECE_NAMES = list(TETROMINOES.keys())

COLORS = {
    'I': (0, 255, 255),
    'O': (255, 255, 0),
    'T': (128, 0, 128),
    'S': (0, 255, 0),
    'Z': (255, 0, 0),
    'J': (0, 0, 255),
    'L': (255, 165, 0),
}


class TetrisEnv:

    """

    Tetris environment.

    1. **manual**: Slow progress.

    2. **training mode**: Instant hard-drop.

    """

    def __init__(self, rows=20, cols=10):

        self.rows = rows
        self.cols = cols

        self.reset()


    def reset(self):

        self.board = np.zeros( (self.rows, self.cols), dtype=np.float32)

        # init variables
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.current_piece = self._random_piece()
        self.next_piece = self._random_piece()

        # manual play state
        self.piece_row = 0
        self.piece_col = 0
        self.piece_rotation = 0
        self.piece_active = False

        return self._get_state()

    def _random_piece(self):

        return random.choice(PIECE_NAMES)

    def _get_rotations(self, piece_name):

        return TETROMINOES[piece_name]

    def _current_shape(self):

        """
        Return the block offsets for the current rotation.
        """

        return self._get_rotations(self.current_piece)[self.piece_rotation]


    def _is_valid_position(self, shape, row, col):

        """
        Check if placing shape at (row, col) is inside bounds and empty.
        """

        for dr, dc in shape:

            r, c = row + dr, col + dc

            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return False
            if self.board[r][c]:
                return False
            
        return True

    # manual play methods

    def spawn_piece(self):

        """
            spawn the current piece at the top-center of the board
            returns False if the piece cannot be spawned (game over)
        """
        
        self.piece_rotation = 0

        shape = self._current_shape()
        min_col = min(c for _, c in shape)
        max_col = max(c for _, c in shape)

        self.piece_col = (self.cols - (max_col - min_col + 1)) // 2 - min_col
        self.piece_row = 0

        if not self._is_valid_position(shape, self.piece_row, self.piece_col):

            self.game_over = True
            self.piece_active = False

            return False

        self.piece_active = True
        return True

    def move_left(self):
        if not self.piece_active:
            return False
        shape = self._current_shape()
        if self._is_valid_position(shape, self.piece_row, self.piece_col - 1):
            self.piece_col -= 1
            return True
        return False

    def move_right(self):
        if not self.piece_active:
            return False
        shape = self._current_shape()
        if self._is_valid_position(shape, self.piece_row, self.piece_col + 1):
            self.piece_col += 1
            return True
        return False

    def rotate(self):
        """Rotate the piece clockwise. Tries the new rotation; reverts if invalid."""
        if not self.piece_active:
            return False
        rotations = self._get_rotations(self.current_piece)
        new_rot = (self.piece_rotation + 1) % len(rotations)
        new_shape = rotations[new_rot]
        if self._is_valid_position(new_shape, self.piece_row, self.piece_col):
            self.piece_rotation = new_rot
            return True
        # Simple wall-kick: try shifting left or right by 1
        for nudge in [-1, 1, -2, 2]:
            if self._is_valid_position(new_shape, self.piece_row, self.piece_col + nudge):
                self.piece_col += nudge
                self.piece_rotation = new_rot
                return True
        return False

    def tick(self):
        """
        Move the active piece down by one row (gravity).
        If it can't move down, lock it in place, clear lines, and advance.
        Returns: (reward, done, info)
        """
        if not self.piece_active:
            return 0, self.game_over, {"score": self.score, "lines_cleared": self.lines_cleared}

        shape = self._current_shape()

        if self._is_valid_position(shape, self.piece_row + 1, self.piece_col):
            self.piece_row += 1
            return 0, False, {"score": self.score, "lines_cleared": self.lines_cleared}
        else:
            return self._lock_and_advance()

    def soft_drop(self):
        """Move piece down one row immediately (player pressed down)."""
        return self.tick()

    def hard_drop_live(self):
        """
        Instantly drop the active piece to the lowest valid row and lock it.
        Returns: (reward, done, info)
        """
        if not self.piece_active:
            return 0, self.game_over, {"score": self.score, "lines_cleared": self.lines_cleared}

        shape = self._current_shape()
        while self._is_valid_position(shape, self.piece_row + 1, self.piece_col):
            self.piece_row += 1

        return self._lock_and_advance()

    def _lock_and_advance(self):
        """Lock the current piece onto the board, clear lines, spawn next."""
        shape = self._current_shape()
        for dr, dc in shape:
            r, c = self.piece_row + dr, self.piece_col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.board[r][c] = 1.0

        self.piece_active = False
        cleared = self._clear_lines()

        line_rewards = {
            0: 0, 
            1: 40, 
            2: 100, 
            3: 300, 
            4: 1200
        }

        reward = line_rewards.get(cleared, 1200)

        self.score += reward
        self.lines_cleared += cleared

        # Advance to next piece
        self.current_piece = self.next_piece
        self.next_piece = self._random_piece()

        if not self.spawn_piece():
            reward -= 10
            return reward, True, {"score": self.score, "lines_cleared": self.lines_cleared}

        return reward, False, {"score": self.score, "lines_cleared": self.lines_cleared}

    def get_ghost_row(self):

        """
            return the row where the piece would land if hard-dropped (for preview).
        """

        if not self.piece_active:
            return self.piece_row
        
        shape = self._current_shape()

        row = self.piece_row
        while self._is_valid_position(shape, row + 1, self.piece_col):
            row += 1

        return row

    # line clearing
    def _clear_lines(self):

        full_rows = [i for i in range(self.rows) if all(self.board[i])]

        for row in full_rows:
            self.board = np.delete(self.board, row, axis=0)
            self.board = np.vstack([np.zeros((1, self.cols), dtype=np.float32), self.board])

        return len(full_rows)

    # RL training interface (instant hard-drop)

    def get_possible_actions(self):

        """
            Returns a list of (rotation_index, column) tuples for all valid
            placements of the current piece (used by the RL agent).
        """

        actions = []
        rotations = self._get_rotations(self.current_piece)

        for rot_idx, shape in enumerate(rotations):

            min_col = min(c for _, c in shape)
            max_col = max(c for _, c in shape)
            for col in range(-min_col, self.cols - max_col):
                if self._check_hard_drop_valid(shape, col):
                    actions.append((rot_idx, col))

        return actions

    def _check_hard_drop_valid(self, shape, col):

        for dr, dc in shape:

            c = col + dc
            if c < 0 or c >= self.cols:
                return False
            
        return True

    def _get_drop_row(self, shape, col):

        last_valid = -1

        for row in range(self.rows):
            if self._is_valid_position(shape, row, col):

                last_valid = row

            else:

                break

        return last_valid

    def step(self, action):

        """
            RL training step: instantly hard-drop the piece at (rotation, column)
            returns: (state, reward, done, info)
        """

        rot_idx, col = action

        shape = self._get_rotations(self.current_piece)[rot_idx]

        drop_row = self._get_drop_row(shape, col)

        if drop_row < 0:

            self.game_over = True

            return self._get_state(), -10.0, True, {"score": self.score}

        # lock piece in place
        for dr, dc in shape:

            r, c = drop_row + dr, col + dc
            self.board[r][c] = 1.0

        cleared = self._clear_lines()

        line_rewards = {
            0: 0, 
            1: 40, 
            2: 100, 
            3: 300, 
            4: 1200
        }

        reward = line_rewards.get(cleared, 1200) # upper bound for more than 4 lines cleared

        self.score += reward
        self.lines_cleared += cleared

        heights = self._get_column_heights()
        reward -= max(heights) * 0.5

        self.current_piece = self.next_piece
        self.next_piece = self._random_piece()

        if not self.get_possible_actions():

            self.game_over = True
            reward -= 10.0

        info_ = {
            "score": self.score,
            "lines_cleared": self.lines_cleared,
        }

        return self._get_state(), reward, self.game_over, info_

    # state features for the RL agent

    def _get_column_heights(self):

        heights = []
        for c in range(self.cols):
            h = 0
            for r in range(self.rows):
                if self.board[r][c]:
                    h = self.rows - r
                    break
            heights.append(h)

        return heights

    def _count_holes(self):

        holes = 0
        for c in range(self.cols):
            found = False
            for r in range(self.rows):
                if self.board[r][c]:
                    found = True
                elif found:
                    holes += 1

        return holes

    def _bumpiness(self):

        heights = self._get_column_heights()

        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    def _complete_lines(self):

        return sum(1 for r in range(self.rows) if all(self.board[r]))

    def _get_state(self):

        heights = self._get_column_heights()
        holes = self._count_holes()
        bumpiness = self._bumpiness()
        total_height = sum(heights)
        complete = self._complete_lines()

        return np.array(heights + [holes, bumpiness, total_height, complete], dtype=np.float32)

    def get_state_for_action(self, action):

        """
            simulate an action and return the resulting state (without modifying the board)
        """

        rot_idx, col = action
        shape = self._get_rotations(self.current_piece)[rot_idx]
        board_copy = self.board.copy()
        drop_row = self._get_drop_row(shape, col)

        if drop_row < 0:
            return None
        
        for dr, dc in shape:
            r, c = drop_row + dr, col + dc
            board_copy[r][c] = 1.0

        full_rows = [i for i in range(self.rows) if all(board_copy[i])]

        for row in full_rows:
            board_copy = np.delete(board_copy, row, axis=0)
            board_copy = np.vstack([np.zeros((1, self.cols), dtype=np.float32), board_copy])

        heights = []
        for c in range(self.cols):
            h = 0
            for r in range(self.rows):
                if board_copy[r][c]:
                    h = self.rows - r
                    break
            heights.append(h)
            
        holes = 0
        for c in range(self.cols):
            found = False
            for r in range(self.rows):
                if board_copy[r][c]:
                    found = True
                elif found:
                    holes += 1
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
        total_height = sum(heights)
        complete = sum(1 for r in range(self.rows) if all(board_copy[r]))
        return np.array(heights + [holes, bumpiness, total_height, complete], dtype=np.float32)

    @property
    def state_size(self):
        return self.cols + 4