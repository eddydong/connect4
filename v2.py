import random

debug = False

board0 = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]]

board0 = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,2,1,1,1,2,0],
        [0,2,1,2,1,2,0]]

think_depth = 5 # Depth for the minimax-like algorithm
row_n = len(board0)
col_n = len(board0[0])
dir = [[1,0], [1,1], [0,1], [-1,1],
       [-1,0], [-1,-1],[0,-1], [1,-1]]

# Function to check if there is a winning condition on the board
def isWin(board, side):
    for c in range(col_n):
        for r in range(row_n):
            if board[r][c] == 0:
                continue
            for i in range(len(dir)):
                conn4 = True
                for step in range(4):
                    cc = c + dir[i][0] * step
                    rr = r + dir[i][1] * step
                    if (cc not in range(0, col_n)) or (rr not in range(0, row_n)) or board[rr][cc]!=side:
                        conn4 = False
                        break
                if conn4:
                    return True
    return False

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

# Function to calculate the best move for a player using a minimax-like algorithm
def maxV(board, side, depth):
    board=deepCopy2D(board)
    score = [0] * col_n
    if depth == 0:
        return randMax(score)
    for c in range(col_n):
        full = True
        for r in range(row_n-1, -1, -1):
            if board[r][c] == 0:
                board[r][c] = side
                if isWin(board, side):
                    score[c] = 1
                    return c, score[c]
                else:
                    score[c] -= 0.9 * maxV(board, 2-(side-1), depth-1)[1]
                board[r][c] = 0
                full = False
                break
        if full:
            score[c] = float("-inf")
    if debug:
        show(board)
        print("X" if side==1 else "O", score)
    return randMax(score)

# Function to make a move on the board
def go(side, col):
    for r in range(row_n-1, -1, -1):
        if board0[r][col] == 0:
            board0[r][col] = side
            return

# Function to create a deep copy of a 2D list (board)
def deepCopy2D(o):
    return [row[:] for row in o]

# Randomly select the maximum value (index, value) from a list
def randMax(l):
    max_value = max(l)
    max_indices = [i for i, val in enumerate(l) if val == max_value]
    pick = random.choice(max_indices)
    return pick, l[pick]

# Initialize the board and start the AI vs AI game
def AvA(verbose=False):
    global board0
    board0 = [[0]*7 for _ in range(6)]  # Reset the board
    while (not isWin(board0, 1)) and (not isWin(board0, 2)):
        col = maxV(board0, 2, think_depth)
        print("Best move for O:", col[0], "@ score:", col[1]) if verbose else None
        go(2, col[0])
        show(board0) if verbose else None
        if isWin(board0, 2):
            print("O wins!") if verbose else None
            return 2
            break

        col = maxV(board0, 1, think_depth)
        print("Best move for X:", col[0], "@ score:", col[1]) if verbose else None
        go(1, col[0])
        show(board0) if verbose else None
        if isWin(board0, 1):
            print("X wins!") if verbose else None
            return 1
            break

        if all(board0[0][c] != 0 for c in range(col_n)):
            print("It's a draw!") if verbose else None
            return 0
            break

# Initialize the board and start the People vs AI game
def PvA():
    global board0
    # board0 = [[0]*7 for _ in range(6)]  # Reset the board
    show(board0)
    while (not isWin(board0, 1)) and (not isWin(board0, 2)):
        print("AI thinking...")
        col = maxV(board0, 2, think_depth)
        go(2, col[0])
        print("AI move:", col[0], "@ score:", col[1])
        show(board0)
        if isWin(board0, 2):
            print("O wins!")
            break

        sel = input("Your move (0-6): ")
        go(1, int(sel))
        show(board0)
        if isWin(board0, 1):
            print("X wins!")
            break

        if all(board0[0][c] != 0 for c in range(col_n)):
            print("It's a draw!")
            break

def benchMark(n):
    game_count = 0
    win_1 = 0
    win_2 = 0
    for i in range(n):
        r = AvA()
        if r == 1:
            win_1 += 1
        elif r == 2:
            win_2 += 1
        game_count += 1
        print(str(round(game_count / n * 100, 0))+"%" , end='\r')
    print("Total games:", game_count, ", X wins:", win_1, ", O wins:", win_2, ", Draws:", game_count - win_1 - win_2)

PvA()

# benchMark(100)

# show(board0)
# col = maxV(board0, 1, think_depth)
# print("Best move:", col[0], "@ score:", col[1])
# go(1, col[0])
# show(board0)