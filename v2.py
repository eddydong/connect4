import random
import time

debug = False

# Initial board to all zeros
# Commented out if you want to use board0
board = [[0] * 7 for _ in range(6)]

row_n = len(board)
col_n = len(board[0])

gameProgress = 0 # starting from 1 ending at 42
prethink_depth = 3  # Depth for pre-thinking in AI
current_player = 1

# player 1 is always X, player 2 is always O
players = {1: {"type":"HUMAN", "name": "You"}, 
           2: {"type":"AI", "name": "AI", "level": 10}} 

# Function to check board status
# -1: No win, 0: Draw, 1: Player 1 wins, 2: Player 2 wins
def checkStatus(board):
    # Check if top row is full (Draw)
    if all(board[0][c] != 0 for c in range(col_n)):
        return 0  # Draw
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
    global gameProgress
    # count stones on the board
    gameProgress = sum(board[r][c] != 0 for r in range(row_n) for c in range(col_n))
    for c in range(col_n):
        print(c, end=' ')
    print("... "+str(gameProgress)+"/42")
    for r in range(row_n):
        for c in range(col_n):
            if board[r][c] == 0:
                print('.', end=' ')
            elif board[r][c] == 1:
                print('X', end=' ')
            else:
                print('O', end=' ')
        print()
    print()

# Function to calculate the best move for a player using a minimax-like algorithm
def maxV(board, side, depth, alpha=float("-inf"), beta=float("inf")):
    
    # Make a deep copy of the board to avoid modifying the original
    board=deepCopy2D(board)

    # Initialize variables for tracking the best score and column
    maxScore = float("-inf")
    maxCol = -1

    # Check if max depth is reached 
    if depth == 0:
        return 0, 0
    
    # Heuristic: Iterate columns from center to edge with randomization
    center = col_n // 2
    col_order = [center]
    for offset in range(1, center + 1):
        cols = []
        if center - offset >= 0:
            cols.append(center - offset)
        if center + offset < col_n:
            cols.append(center + offset)
        random.shuffle(cols)
        col_order.extend(cols)

    # Iterate through columns in the randomized order
    for c in col_order:
        score = None # Initialize score to None for each column
        for r in range(row_n-1, -1, -1): # Check from bottom to top
            # Check if the column is not full
            if board[r][c] == 0:
                score = 0
                board[r][c] = side
                winSide = checkStatus(board)
                if winSide == side:
                    score = 1
                else:
                    # Recursive call with swapped and negated alpha/beta
                    score = -maxV(board, 2-(side-1), depth-1, -beta, -alpha)[1]
                board[r][c] = 0
                break

        if debug and depth==3:
            print("X" if side==1 else "O","col:"+str(c), "score:", score)

        # update maxScore and maxCol if a better score is found
        if score is not None and maxScore < score:
            maxScore = score
            maxCol = c

        # Correct alpha-beta pruning
        alpha = max(alpha, maxScore)
        if alpha >= beta:
            break # Prune

    # debug output
    if debug and depth==3:
        print("depth="+str(depth)+":", "X" if side==1 else "O", "getting", maxScore, "at", maxCol)
        show(board)

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
            # AI's turn to make a move
            start_time = time.time()
            # Check for immediate winning move from opponent perspective
            col = maxV(board, 3-current_player, prethink_depth)
            # If any, shallow thinking to defend without deep thinking
            if col[1] == 1 or gameProgress < 3:
                print("AI shallow-thinking("+str(prethink_depth)+")...")
                col = maxV(board, current_player, prethink_depth+1)
            else: # If not, perform deep thinking
                print("AI deep-thinking("+str(players[current_player]["level"])+")...")
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

        show(board) # Print the board after each move
        # Check the status of the game
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