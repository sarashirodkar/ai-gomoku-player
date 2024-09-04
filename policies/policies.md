## Code Structure

- `gomoku.py`: This module contains the implementation of the Gomoku game, including methods to list valid actions for every state, perform an action in a given state, and calculate the score in a game state.
- `compete.py`: This module runs a full game between two players. Each player can be controlled by a human or various automated policies.
- `performance.py`: This module runs several games between your AI and the baseline policy. The plots at the end visualize the final scores and run times of each AI. It saves the results in a file named `perf.pkl`.
- `policies/`: The modules in this sub-directory include various policies that can be selected for each player.
  - `human.py`: This policy is human-controlled.
  - `random.py`: This policy chooses actions uniformly at random.
  - `minimax.py`: This policy chooses actions using an augmented Minimax Search.
  - `submission.py`: This policy will use your AI implementation.

## API Description

The max player uses the symbol “x” and the min player uses the symbol “o.” Internally,the board is represented by a 3D NumPy array with shape `(3, board_size, board_size)` and entries that are either 0 or 1. The leading dimension corresponds to the status of each position: whether it is empty, occupied by the min player, or occupied by the max player. Specifically, if position `(r,c)` is empty, then `state[0,r,c] == 1`, otherwise `state[0,r,c] == 0`.
Similarly, if the min player has a mark at position `(r,c)`, then `state[1,r,c] == 1`, otherwise `state[1,r,c] == 0`. Likewise for the max player, in `state[2,r,c]`.
One example 3x3 state is as follows:
Text representation state
[0,:,:] state[1,:,:] state[2,:,:]
..x 110 000 001
.o. 101 010 000
x.. 011 000 100

A Gomoku game is parameterized by board size and a win size. The board is a square and board size is the side length of the square. The win size is the number of marks that must be placed in a row to win.
The game score rewards winners for finishing more quickly. The sign of the score indicates whether min or max won. The magnitude of the score is the number of empty positions left at 
the end of the game (plus 1).Actions are represented by a tuple `(r,c)` which indicates that the current player will put their mark at row `r` and column `c`. Row and column indices are zero-based.
The comments in `gomoku.py` also provide more information on the available helper methods on gomoku states, which return things like current player, score, valid actions, or the new 
state after an action is performed. For examples of an AI using the gomoku helpers, check the code and comments in `policies/minimax.py`.

## Implementation Guidelines

Each policy, including the one you must implement, is a Python class with the following methods:

- `__init__(self, board_size, win_size)`: This method initializes the policy for the given Gomoku game parameters. This initialization can also optionally do things like load lookup
tables or optimized neural network weights.
- `__call__(self, state)`: This method must return a valid action in the given state. This is the entry point to your AI method.

You can run a game between two policies by running `compete.py` on the command line with options to specify board size, win size, and each player’s policy. For example,
the command: python compete.py -b 15 -w 5 -x Human -o Minimax will run a game on a 15x15 board with 5 in a row to win, where you control the max player, and the min player
selects actions randomly. For each policy you must write the exact name of the policy class (case-sensitive). The available policies are `Human`, `Random`, `Minimax`, and 
`Submission` (which is yours). Press Ctrl-C, or a similar command in your operating system, to interrupt the script at any time and terminate early before the game is over.



