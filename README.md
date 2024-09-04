# AI Game Strategy using Minimax Algorithm

## Introduction

Our AI employs the minimax algorithm with alpha-beta pruning [1] to make strategic decisions in the game. It evaluates the game state by considering winning patterns and board positions, maximizing its chances of winning while obstructing the opponent. We utilized correlation matrices and utility functions to obtain potential winning patterns and mathematically optimized the minimax algorithm by recursively exploring future states.

## Implementation: submission.py

The `Submission` class initializes with parameters: `board_size`, `win_size`, and an optional `max_depth` for the minimax algorithm. The main entry point is the `call` method, invoking the minimax function to determine the best move. Minimax explores potential moves recursively [2], considering player and opponent turns, with alpha-beta pruning optimizing the search. The `evaluate_best_move` method identifies the best move by prioritizing winning patterns and blocking opponent victories using the correlation matrix. The evaluation function incorporates the following strategies:

1. **Correlation Matrix Analysis:**  
   Examines the correlation matrix to identify winning patterns where the agent can win on its next move and block opponent wins. The correlation matrix (`state.corr`) captures potential winning configurations.

2. **Utility Functions used for Decision-Making:**
  - `minimal_path_to_game_over_state`: Calculates the minimal path to a game-over state by considering the remaining empty cells, the opponent's potential wins, and the current turn.
  - `evaluate_best_move`: Queries the correlation matrix to identify potential winning moves for both players.
  - `find_best_empty_board_cells`: Identifies optimal empty cells on the board using winning patterns (0 for upward, 1 for downward, 2 for diagonal, and 3 for anti-diagonal). It introduces randomness through move order scrambling to enhance adaptability in decision-making.

## Performance

The AI demonstrated a win rate of approximately 67%, showcasing strategic decision-making and competitiveness against Minimax. Turn times for both the AI and Minimax remained reasonable. The scores indicate successful strategic moves in multiple repetitions. To enhance future performance, we can consider adjusting randomization, dynamically modifying `max_depth`, tuning hyperparameters, and implementing opponent modeling. We faced challenges in balancing `max_depth` and refining move evaluation logic, which we addressed through successful parameter tuning, optimizing exploration and exploitation for effective decision-making to achieve better results.

## Bibliography

1. Rijul Nasa, Rishabh Didwania, Shubhranil Maji, Vipul Kumar (2018). Alpha-Beta Pruning in Mini-Max Algorithm – An Optimized Approach for a Connect-4 Game. International Research Journal of Engineering and Technology (IRJET). Volume: 05 Issue: 04

2. Kuan Liang Tan, Chin Hiong Tan, Kay Chen Tan and Arthur Tay, “Adaptive Game AI for Gomoku”, Proceedings of the 4th International Conference on Autonomous Robots and Agents, Feb 10-12, 2009

3. Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).

4. Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.
