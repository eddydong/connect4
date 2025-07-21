import random
import time

debug = False

board0 = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,2,1,1,1,2,0],
        [0,2,1,2,1,2,0]]

board1 = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,2,0,0,0],
        [0,1,1,1,2,0,0]]

# Initial board to all zeros
# Commented out if you want to use board0
board = [[0] * 7 for _ in range(6)]

board = board  # Use the predefined board0 for testing

row_n = len(board)
col_n = len(board[0])

# player 1 is always X, player 2 is always O
players = {1: {"type":"HUMAN", "name": "You", "level":0}, 
           2: {"type":"AI", "name": "AI_6", "level":12},
           22: {"type":"AI", "name": "AI_5", "level":5}} 
current_player = 1

# Function to check board status
# -1: No win, 0: Draw, 1: Player 1 wins, 2: Player 2 wins
def checkStatus(board):
    # Check if top row is full (Draw)
    if all(board[0][c] != 0 for c in range(col_n)):
        return 0
    # Horizontal (rows 0-5, cols 0-3)
    for r in range(row_n):
        for c in range(col_n - 3):
            if board[r][c] == 1 and board[r][c+1] == 1 and board[r][c+2] == 1 and board[r][c+3] == 1:
                return 1
            elif board[r][c] == 2 and board[r][c+1] == 2 and board[r][c+2] == 2 and board[r][c+3] == 2:
                return 2
    # Vertical (rows 0-2, cols 0-6)
    for c in range(col_n):
        for r in range(row_n - 3):
            if board[r][c] == 1 and board[r+1][c] == 1 and board[r+2][c] == 1 and board[r+3][c] == 1:
                return 1
            elif board[r][c] == 2 and board[r+1][c] == 2 and board[r+2][c] == 2 and board[r+3][c] == 2:
                return 2
    # Diagonal / (bottom left to top right)
    for r in range(row_n - 3):
        for c in range(col_n - 3):
            if board[r][c] == 1 and board[r+1][c+1] == 1 and board[r+2][c+2] == 1 and board[r+3][c+3] == 1:
                return 1
            elif board[r][c] == 2 and board[r+1][c+1] == 2 and board[r+2][c+2] == 2 and board[r+3][c+3] == 2:
                return 2
    # Diagonal \ (top left to bottom right)
    for r in range(3, row_n):
        for c in range(col_n - 3):
            if board[r][c] == 1 and board[r-1][c+1] == 1 and board[r-2][c+2] == 1 and board[r-3][c+3] == 1:
                return 1
            elif board[r][c] == 2 and board[r-1][c+1] == 2 and board[r-2][c+2] == 2 and board[r-3][c+3] == 2:
                return 2
    return -1

# Print the board, where 1 is X, 2 is O
def show(board):
    for c in range(col_n):
        print(c, end=' ')
    print()
    for r in range(row_n):
        for c in range(col_n):
            if board[r][c] == 0:
                print('.', end=' ')
            elif board[r][c] == 1:
                print('X', end=' ')
            else:
                print('O', end=' ')
        print()

# Function to find if the opponent can win in the next move
def dangerous(board, side):
    # Return the column where the opponent can win in the next move, or -1 if none
    for c in range(col_n):
        for r in range(row_n-1, -1, -1):
            if board[r][c] == 0:
                board[r][c] = 3-side
                if checkStatus(board) == 3-side:
                    board[r][c] = 0  # Undo move
                    return c
                board[r][c] = 0  # Undo move
                break
    return -1

# Function to calculate the best move for a player using a minimax-like algorithm
def maxV(board, side, depth, alpha=float("-inf"), beta=float("inf")):
    isDanger = dangerous(board, side)
    if isDanger != -1:
        return isDanger, -1
    board=deepCopy2D(board)
    maxScore = float("-inf")
    maxCol = -1
    if depth == 0:
        return 0, 0
    # Heuristic: Iterate columns from center to edge
    center = col_n // 2
    col_order = [center]
    for offset in range(1, center + 1):
        if center - offset >= 0:
            col_order.append(center - offset)
        if center + offset < col_n:
            col_order.append(center + offset)
    for c in col_order:
        score = None
        for r in range(row_n-1, -1, -1):
            if board[r][c] == 0:
                score = 0
                board[r][c] = side
                winSide = checkStatus(board)
                if winSide == side:
                    score = 1
                    board[r][c] = 0 # Undo move before returning
                    return c, score
                else:
                    # Recursive call with swapped and negated alpha/beta
                    score = -maxV(board, 2-(side-1), depth-1, -beta, -alpha)[1]
                board[r][c] = 0
                break
        if score is not None and maxScore < score:
            maxScore = score
            maxCol = c

        # Correct alpha-beta pruning
        alpha = max(alpha, maxScore)
        if alpha >= beta:
            break # Prune

    if debug:
        show(board)
        print("X" if side==1 else "O", maxScore, "at", maxCol)
    return maxCol, maxScore

# Function to make a move on the board
def go(side, col):
    for r in range(row_n-1, -1, -1):
        if board[r][col] == 0:
            board[r][col] = side
            return 1
    print("Column is full, cannot make a move.")
    return 0

# Function to create a deep copy of a 2D list (board)
def deepCopy2D(o):
    return [row[:] for row in o]

# Randomly select the maximum value (index, value) from a list
# where l is a list of scores, like [1, 2, 3, ...]
def randMax(l):
    max_value = max(l)
    max_indices = [i for i, val in enumerate(l) if val == max_value]
    pick = random.choice(max_indices)
    return pick, l[pick]

# Initialize the board and start the People vs AI game
def gameplay():
    global current_player
    show(board)
    gameOn = True
    while (gameOn):
        if (players[current_player]["type"]=="AI"):
            print("AI thinking...")
            start_time = time.time()
            col = maxV(board, current_player, players[current_player]["level"])
            elapsed = time.time() - start_time
            print(players[current_player]["name"], "move:", col[0], "@ score:", col[1], "time used: {:.3f}s".format(elapsed))
            go(current_player, col[0])
        elif (players[current_player]["type"]=="HUMAN"):
            # Get user input for the move, handling invalid input
            sel = None
            while sel is None:
                try:
                    sel = int(input("Input your move (0-6): "))
                    if sel < 0 or sel >= col_n or board[0][sel] != 0:
                        print("Invalid move.")
                        sel = None
                except ValueError:
                    print("Invalid input!")
            go(current_player, sel)
        else:
            print("Unknown player type, skipping turn.")

        show(board)
        status = checkStatus(board)
        if status > 0:
            print(players[status]["name"], "wins!")
            gameOn = False
        elif status == 0:
            print("It's a draw!")
            gameOn = False
        else:
            current_player = 3 - current_player  # Switch player

gameplay()
