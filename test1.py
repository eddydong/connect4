import numpy as np
import random
import pickle

# Connect-4 Environment
class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def make_move(self, col):
        if col not in self.get_valid_moves():
            return None, -10, True  # Invalid move penalty
        for row in range(self.rows-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                break
        if self.check_win(self.current_player):
            return self.get_state(), 10, True
        if len(self.get_valid_moves()) == 0:
            return self.get_state(), 0, True  # Draw
        self.current_player = 3 - self.current_player
        return self.get_state(), 0, False

    def check_win(self, player):
        for r in range(self.rows):
            for c in range(self.cols):
                if c <= self.cols-4 and all(self.board[r][c:c+4] == player):
                    return True
                if r <= self.rows-4 and all(self.board[r:r+4, c] == player):
                    return True
                if r <= self.rows-4 and c <= self.cols-4:
                    if all(self.board[r+i][c+i] == player for i in range(4)):
                        return True
                if r >= 3 and c <= self.cols-4:
                    if all(self.board[r-i][c+i] == player for i in range(4)):
                        return True
        return False

    def get_state(self):
        # Simplified state: count of 2, 3, and 4-in-a-row opportunities for each player
        state = []
        for player in [1, 2]:
            twos, threes, fours = 0, 0, 0
            # Horizontal
            for r in range(self.rows):
                for c in range(self.cols-3):
                    line = self.board[r, c:c+4]
                    if np.sum(line == player) == 2 and np.sum(line == 0) == 2:
                        twos += 1
                    if np.sum(line == player) == 3 and np.sum(line == 0) == 1:
                        threes += 1
            # Vertical
            for c in range(self.cols):
                for r in range(self.rows-3):
                    line = self.board[r:r+4, c]
                    if np.sum(line == player) == 2 and np.sum(line == 0) == 2:
                        twos += 1
                    if np.sum(line == player) == 3 and np.sum(line == 0) == 1:
                        threes += 1
            # Diagonals
            for r in range(self.rows-3):
                for c in range(self.cols-3):
                    line = [self.board[r+i][c+i] for i in range(4)]
                    if line.count(player) == 2 and line.count(0) == 2:
                        twos += 1
                    if line.count(player) == 3 and line.count(0) == 1:
                        threes += 1
            for r in range(3, self.rows):
                for c in range(self.cols-3):
                    line = [self.board[r-i][c+i] for i in range(4)]
                    if line.count(player) == 2 and line.count(0) == 2:
                        twos += 1
                    if line.count(player) == 3 and line.count(0) == 1:
                        threes += 1
            state.extend([twos, threes, fours])
        return tuple(state)  # Hashable state

    def display_board(self):
        print("\n  0 1 2 3 4 5 6")
        for row in self.board:
            print("|", end=" ")
            for cell in row:
                if cell == 0:
                    print(".", end=" ")
                elif cell == 1:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print("|")
        print("-" * 15)

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]

    def choose_action(self, state, valid_moves, explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice(valid_moves)
        q_values = [self.get_q_value(state, a) for a in valid_moves]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_moves, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, next_valid_moves):
        current_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in next_valid_moves], default=0)
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("No saved Q-table found, starting fresh.")

# Simple opponent for training
def random_opponent(env, valid_moves):
    # 80% random, 20% block/win if possible
    if random.random() < 0.2:
        for col in valid_moves:
            temp_board = env.board.copy()
            for r in range(env.rows-1, -1, -1):
                if temp_board[r][col] == 0:
                    temp_board[r][col] = env.current_player
                    break
            if env.check_win(env.current_player):
                return col
            temp_board[r][col] = 3 - env.current_player
            if env.check_win(3 - env.current_player):
                return col
    return random.choice(valid_moves)

# Training Loop
def train_agent(episodes=10000):
    env = Connect4()
    agent = QLearningAgent(alpha=0.2, gamma=0.95, epsilon=0.2)
    wins = 0
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                break
            if env.current_player == 1:  # Agent's turn
                action = agent.choose_action(state, valid_moves)
            else:  # Opponent's turn
                action = random_opponent(env, valid_moves)
            
            next_state, reward, done = env.make_move(action)
            if next_state is None:
                continue
            
            if env.current_player == 1:  # Update Q-table only for agent's moves
                next_valid_moves = env.get_valid_moves() if not done else []
                agent.update_q_table(state, action, reward, next_state, next_valid_moves)
            
            state = next_state
            if done and reward == 10 and env.current_player == 1:
                wins += 1
        
        agent.epsilon = max(0.01, agent.epsilon * 0.999)  # Slower decay
        
        if episode % 1000 == 0:
            print(f"Episode {episode}, Epsilon: {agent.epsilon:.3f}, Win rate: {wins/(episode+1):.3f}")
    
    agent.save_q_table()
    return agent

# Play against the trained agent
def play_game(agent):
    env = Connect4()
    state = env.reset()
    done = False
    human_player = 1
    
    print("Welcome to Connect-4! You are 'X', the agent is 'O'. Enter column (0-6) to play.")
    
    while not done:
        env.display_board()
        valid_moves = env.get_valid_moves()
        
        if env.current_player == human_player:
            try:
                col = int(input(f"Player X, choose a column (0-6): "))
                if col not in valid_moves:
                    print("Invalid move! Try again.")
                    continue
            except ValueError:
                print("Please enter a number between 0 and 6.")
                continue
        else:
            col = agent.choose_action(state, valid_moves, explore=False)
            print(f"Agent chooses column {col}")
        
        next_state, reward, done = env.make_move(col)
        if next_state is None:
            if env.current_player == human_player:
                print("Invalid move! Try again.")
                continue
            else:
                print("Agent made an invalid move (shouldn't happen).")
                break
        
        state = next_state
        
        if done:
            env.display_board()
            if reward == 10:
                winner = "X" if env.current_player == human_player else "O"
                print(f"Game Over! {winner} wins!")
            else:
                print("Game Over! It's a draw!")

if __name__ == "__main__":
    print("Training the agent...")
    agent = train_agent(10000)
    print("Training complete! Starting game...")
    play_game(agent)