import numpy as np
import random
import pygame
import math
from time import sleep, time

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
PLAYER = 1
CPU = -1
EMPTY = 0
PLAYER_PIECE = 1
CPU_PIECE = -1
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WINDOW_LENGTH = 4
WINNING_SCORE = 10000000000000

class Connect4UI:
    def __init__(self, width=COLUMN_COUNT*SQUARESIZE, height=(ROW_COUNT+1)*SQUARESIZE):
        pygame.init()
        self.width = width
        self.height = height
        self.size = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.size)
        self.font = pygame.font.SysFont("monospace", 75)

    def draw_board(self, board):
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(self.screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(self.screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if board[r][c] == PLAYER_PIECE:
                    pygame.draw.circle(self.screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), self.height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
                elif board[r][c] == CPU_PIECE:
                    pygame.draw.circle(self.screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2), self.height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)

        pygame.display.update()
        sleep(0.2)
        
    def display_winner(self, winner):
        if winner == PLAYER:
            label = self.font.render("Player wins!!", 1, RED)
        elif winner == CPU:
            label = self.font.render("Computer wins!!", 1, YELLOW)
        else:
            label = self.font.render("It's a draw!!", 1, BLUE)
        self.screen.blit(label, (40, 10))
        pygame.display.update()
        sleep(5)

class Connect4Game:
    def __init__(self, ui, minimax_depth=1, prune=True):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))
        self.ui = Connect4UI() if ui else None
        self.minimax_depth = minimax_depth
        self.prune = prune
        self.current_turn = random.choice([1, -1])
        
    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece

    def get_next_open_row(self, board, col):
        for r in range(ROW_COUNT):
            if board[r][col] == 0:
                return r

    def print_board(self, board):
        print(np.flip(board, 0))

    def winning_move(self, board, piece):
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if all(board[r][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if all(board[r+i][c] == piece for i in range(WINDOW_LENGTH)):
                    return True

        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if all(board[r+i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if all(board[r-i][c+i] == piece for i in range(WINDOW_LENGTH)):
                    return True

        return False
    
    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = PLAYER_PIECE
        if piece == PLAYER_PIECE:
            opp_piece = CPU_PIECE
        
        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2
        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 4
        
        return score
    
    def score_position(self, board, piece):
        score = 0  
        center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
        center_count = center_array.count(piece)
        score += center_count * 3
    
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(COLUMN_COUNT-3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for c in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(ROW_COUNT-3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)
                
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)
    
        return score
    
    def is_terminal_node(self, board):
        return self.winning_move(board, PLAYER_PIECE) or self.winning_move(board, CPU_PIECE) or len(self.get_valid_locations(board)) == 0
    
    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if board[ROW_COUNT-1][col] == 0:
                valid_locations.append(col)
        return valid_locations
    
    def heuristic(self, board, piece):
        if self.is_terminal_node(board):
            if self.winning_move(board, piece):
                return WINNING_SCORE
            elif self.winning_move(board, -piece):
                return -WINNING_SCORE
            else:
                return 0
        else:
            return self.score_position(board, piece) - self.score_position(board, -piece)

    # Implement minimax algorithm with alpha-beta pruning and depth limiting
    def minimax(self, board, depth, alpha, beta, player):
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board)
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.winning_move(board, CPU_PIECE):
                    return (None, WINNING_SCORE)
                elif self.winning_move(board, PLAYER_PIECE):
                    return (None, -WINNING_SCORE)
                else:
                    return (None, 0)
            else:
                return (None, self.heuristic(board, CPU_PIECE if player == CPU else PLAYER_PIECE))
        
        if player == CPU:
            max_eval = -math.inf
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, CPU_PIECE)
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, PLAYER)[1]
                if new_score > max_eval:
                    max_eval = new_score
                    best_col = col
                if self.prune:
                    alpha = max(alpha, max_eval)
                    if alpha >= beta:
                        break
            return best_col, max_eval

        else:  # Minimizing player
            min_eval = math.inf
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, PLAYER_PIECE)
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, CPU)[1]
                if new_score < min_eval:
                    min_eval = new_score
                    best_col = col
                if self.prune:
                    beta = min(beta, min_eval)
                    if alpha >= beta:
                        break
            return best_col, min_eval
        
    def get_cpu_move(self):
        column, _ = self.minimax(self.board, self.minimax_depth, -math.inf, math.inf, CPU)
        return column
    
    def play(self):
        if self.ui:
            self.ui.draw_board(self.board)
        
        game_over = False
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                
                if self.current_turn == PLAYER:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x_pos = event.pos[0]
                        col = x_pos // SQUARESIZE
                        if self.board[ROW_COUNT-1][col] == 0:
                            row = self.get_next_open_row(self.board, col)
                            self.drop_piece(self.board, row, col, PLAYER_PIECE)
                            if self.winning_move(self.board, PLAYER_PIECE):
                                self.ui.draw_board(self.board)
                                game_over = True
                                self.ui.display_winner(PLAYER)
                            self.current_turn = CPU
                            self.ui.draw_board(self.board)

                elif self.current_turn == CPU and not game_over:
                    pygame.time.wait(500)
                    col = self.get_cpu_move()
                    if self.board[ROW_COUNT-1][col] == 0:
                        row = self.get_next_open_row(self.board, col)
                        self.drop_piece(self.board, row, col, CPU_PIECE)
                        if self.winning_move(self.board, CPU_PIECE):
                            self.ui.draw_board(self.board)
                            game_over = True
                            self.ui.display_winner(CPU)
                        self.current_turn = PLAYER
                        self.ui.draw_board(self.board)
            
            if len(self.get_valid_locations(self.board)) == 0:
                game_over = True
                self.ui.display_winner(None)

# Start the game
if __name__ == "__main__":
    game = Connect4Game(True, 3, True)
    game.play()
