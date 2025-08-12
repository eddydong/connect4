import random
import copy

# Constants for Connect-4
ROWS = 6
COLUMNS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

def create_board():
	return [[EMPTY for _ in range(COLUMNS)] for _ in range(ROWS)]

def valid_moves(board):
	return [c for c in range(COLUMNS) if board[0][c] == EMPTY]

def make_move(board, col, player):
	for row in reversed(range(ROWS)):
		if board[row][col] == EMPTY:
			board[row][col] = player
			return True
	return False

def check_winner(board):
	# Horizontal, vertical, diagonal checks
	for r in range(ROWS):
		for c in range(COLUMNS):
			if board[r][c] == EMPTY:
				continue
			player = board[r][c]
			# Horizontal
			if c <= COLUMNS - 4 and all(board[r][c+i] == player for i in range(4)):
				return player
			# Vertical
			if r <= ROWS - 4 and all(board[r+i][c] == player for i in range(4)):
				return player
			# Diagonal /
			if r >= 3 and c <= COLUMNS - 4 and all(board[r-i][c+i] == player for i in range(4)):
				return player
			# Diagonal \
			if r <= ROWS - 4 and c <= COLUMNS - 4 and all(board[r+i][c+i] == player for i in range(4)):
				return player
	return None

def is_draw(board):
	return all(board[0][c] != EMPTY for c in range(COLUMNS))

class MCTSNode:
	def __init__(self, board, player, parent=None, move=None):
		self.board = copy.deepcopy(board)
		self.player = player
		self.parent = parent
		self.move = move
		self.children = []
		self.visits = 0
		self.wins = 0

	def best_child(self, c_param=1.4):
		choices_weights = [
			(child.wins / child.visits if child.visits > 0 else 0) +
			c_param * ((2 * (self.visits)**0.5) / (child.visits + 1))
			for child in self.children
		]
		return self.children[choices_weights.index(max(choices_weights))]

	def expand(self):
		moves = valid_moves(self.board)
		for move in moves:
			new_board = copy.deepcopy(self.board)
			make_move(new_board, move, self.player)
			child = MCTSNode(new_board, 3 - self.player, parent=self, move=move)
			self.children.append(child)

def random_playout(board, player):
	current_board = copy.deepcopy(board)
	current_player = player
	while True:
		winner = check_winner(current_board)
		if winner:
			return winner
		if is_draw(current_board):
			return 0
		moves = valid_moves(current_board)
		move = random.choice(moves)
		make_move(current_board, move, current_player)
		current_player = 3 - current_player

def mcts_search(root, itermax=1000):
	for _ in range(itermax):
		node = root
		# Selection
		while node.children:
			node = node.best_child()
		# Expansion
		winner = check_winner(node.board)
		if not winner and not is_draw(node.board):
			node.expand()
			if node.children:
				node = random.choice(node.children)
		# Simulation
		result = random_playout(node.board, node.player)
		# Backpropagation
		while node:
			node.visits += 1
			if result == 0:
				node.wins += 0.5  # Draw
			elif result == 3 - node.player:
				node.wins += 1
			node = node.parent

	best = max(root.children, key=lambda c: c.visits)
	return best.move

def print_board(board):
	print("\n 0 1 2 3 4 5 6")
	for row in board:
		print(' '.join(['.' if x == EMPTY else ('X' if x == PLAYER1 else 'O') for x in row]))

def main():
	board = create_board()
	current_player = PLAYER1
	print("Welcome to Connect-4! You are X. AI is O.")
	while True:
		print_board(board)
		winner = check_winner(board)
		if winner:
			print(f"{'You win!' if winner == PLAYER1 else 'AI wins!'}")
			break
		if is_draw(board):
			print("Draw!")
			break
		if current_player == PLAYER1:
			move = None
			while move not in valid_moves(board):
				try:
					move = int(input("Your move (0-6): "))
				except ValueError:
					continue
			make_move(board, move, PLAYER1)
		else:
			print("AI is thinking...")
			root = MCTSNode(board, PLAYER2)
			root.expand()
			ai_move = mcts_search(root, itermax=1000)
			make_move(board, ai_move, PLAYER2)
			print(f"AI plays column {ai_move}")
		current_player = 3 - current_player

if __name__ == "__main__":
	main()
	def __init__(self, board, player, parent=None, move=None):
		self.board = copy.deepcopy(board)
		self.player = player
		self.parent = parent
		self.move = move
		self.children = []
		self.visits = 0
		self.wins = 0

	def expand(self):
		moves = valid_moves(self.board)
		for move in moves:
			new_board = copy.deepcopy(self.board)
			make_move(new_board, move, self.player)
			child = MCTSNode(new_board, 3 - self.player, parent=self, move=move)
			self.children.append(child)

