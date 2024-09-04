import copy
import itertools as it
import random
import numpy as np
from scipy.signal import correlate
import gomoku as gm

class Submission:

    # Initializes the Gomoku-playing agent with the specified board size, win size, and maximum search depth.
    def __init__(self, board_size, win_size, max_depth=4):
        self.board_size = board_size
        self.win_size = win_size
        self.max_depth=max_depth

    # Executes the Submission object as a callable, taking a game state as input and returning the next action based on the algorithm.
    def __call__(self, state):
        _, action = self.minimax(state, depth=self.max_depth, alpha=-np.inf, beta=np.inf)
        if action not in state.valid_actions():
            return state.valid_actions()[0]
        else:
            return action
    
    # The minimax algorithm with alpha-beta pruning to evaluate the game state and choose the optimal move.
    # It explores the game tree up to the specified depth and considers heuristics for move selection.
    def minimax(self, state, depth, alpha, beta):
        score, action = self.evaluate_best_move(state)
        if score != 0:
            return score, action
        if state.is_game_over():
            return state.current_score(), None
        valid_actions = state.valid_actions()
        rank = -state.corr[:, 1:].sum(axis=(0, 1)) - np.random.rand(*state.board.shape[1:])
        rank = rank[state.board[gm.EMPTY] > 0]
        scrambler = np.argsort(rank)
        if depth == 0:
            return state.current_score(), valid_actions[scrambler[0]]
        if self.minimal_path_to_game_over_state(state) > depth:
            return 0, valid_actions[scrambler[0]]
        if state.is_max_turn():
            value = -np.inf
            best_action = None
            for scrambled_action in scrambler:
                action = valid_actions[scrambled_action]
                child_state = state.perform(action)
                current_utility, _ = self.minimax(child_state, depth - 1, alpha, beta)
                if current_utility > value: 
                    value, best_action = current_utility, action
                if value >= beta: 
                    break
                alpha = max(alpha, value)
        else:
            value = +np.inf
            best_action = None
            for scrambled_action in scrambler:
                action = valid_actions[scrambled_action]
                child_state = state.perform(action)
                current_utility, _ = self.minimax(child_state, depth - 1, alpha, beta)
                if current_utility < value: 
                    value, best_action = current_utility, action
                if value <= alpha: 
                    break
                beta = min(beta, value)

        return value, best_action

    # Evaluates the best move based on the given game state by considering winning patterns, opponent's potential winning moves,
    # and strategic positions. Returns a score and the recommended action.
    def evaluate_best_move(self, state):
        current_player = state.current_player()
        player_sign = 1 if current_player == gm.MAX else -1
        summation = state.board[gm.EMPTY].sum()
        correlate = state.corr
        index = np.argwhere((correlate[:, gm.EMPTY] == 1) & (correlate[:, current_player] == state.win_size - 1))
        if index.shape[0] > 0:
            pattern, row, column = index[0]
            action = self.find_best_empty_board_cells(state, row, column, pattern)
            return player_sign * summation, action
        opponent_player = gm.MIN if state.is_max_turn() else gm.MAX
        loss_empties = set()
        index = np.argwhere((correlate[:, gm.EMPTY] == 1) & (correlate[:, opponent_player] == state.win_size - 1))
        for pattern, row, column in index:
            position = self.find_best_empty_board_cells(state, row, column, pattern)
            loss_empties.add(position)        
            if len(loss_empties) > 1:
                score = -player_sign * (summation - 1)
                return score, position
        board_corners = [(0, 0), (0, state.board.shape[0] - 1), 
                (state.board.shape[0] - 1, 0), (state.board.shape[0] - 1, state.board.shape[0] - 1)]
        for corner in board_corners:
            if state.board[gm.EMPTY, corner[0], corner[1]] == 1:
                return 0, corner
        for i in range(state.board.shape[0]):
            for j in range(state.board.shape[1]):
                if state.board[gm.EMPTY, i, j] == 1:
                    return 0, (i, j)
        return 0, None
    
    # Determines the minimal path to a game-over state by assessing available empty cells and potential winning patterns.
    # Returns the minimum number of moves required to reach a game-over state.
    def minimal_path_to_game_over_state(self, state):
        is_max_player_turn = state.is_max_turn()
        empty_count = np.count_nonzero(state.board == gm.EMPTY)
        corr = state.corr
        empty_cells = corr[:, gm.EMPTY]
        min_possible_win = empty_cells + corr[:, gm.MIN]
        max_possible_win = empty_cells + corr[:, gm.MAX]

        # Handle the case where there are no possible wins
        if not np.any(min_possible_win == state.win_size) and not np.any(max_possible_win == state.win_size):
            return empty_count

        min_possible_win = np.where(min_possible_win != state.win_size, np.inf, min_possible_win)
        max_possible_win = np.where(max_possible_win != state.win_size, np.inf, max_possible_win)
        min_turns = 2 * empty_cells - (0 if is_max_player_turn else 1)
        max_turns = 2 * empty_cells - (1 if is_max_player_turn else 0)
        min_turns = np.where(min_turns < 0, np.inf, min_turns)
        max_turns = np.where(max_turns < 0, np.inf, max_turns)

        # Check if the array is non-empty before computing the minimum
        if np.any(min_possible_win == state.win_size):
            min_possible_moves = np.min(min_turns[min_possible_win == state.win_size])
        else:
            min_possible_moves = np.inf

        if np.any(max_possible_win == state.win_size):
            max_possible_moves = np.min(max_turns[max_possible_win == state.win_size])
        else:
            max_possible_moves = np.inf

        if np.isinf(min_possible_moves) and np.isinf(max_possible_moves):
            return empty_count

        least_moves_possible = min(empty_count, min(min_possible_moves, max_possible_moves))
        return least_moves_possible

    # Identifies the best empty cells on the board based on winning patterns.
    # Utilizes specific slices of the game board to find optimal row and column coordinates for a move.
    # Incorporates randomness by considering variations in move order for decision-making adaptability.
    def find_best_empty_board_cells(self, state, row, column, pattern):
        if pattern == 0: 
            return row, column + state.board[gm.EMPTY, row, column:column + state.win_size].argmax()
        if pattern == 1: 
            return row + state.board[gm.EMPTY, row:row + state.win_size, column].argmax(), column
        if pattern == 2: 
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, row + rng, column + rng].argmax()
            return row + offset, column + offset
        if pattern == 3: 
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, row - rng, column + rng].argmax()
            return row - offset, column + offset
        return None
