# Change -- use BFS instead of DFS, as this allows for pruning
# -- Actually, you can't use BFS for pruning. Use DFS. 

from typing import Dict
import numpy as np
import random
import collections

class Connect4Serializer:
    @staticmethod
    def serialize(gs):
        return repr(gs.board)

    @staticmethod
    def deserialize(s):
        return eval(s)

    @staticmethod
    def display(s):
        return s

# Board state + turn?
class Connect4GameState:
    def __init__(self, board, turn):
        self.board = board or None # np.array
        self.turn = turn or None # either X or O

    def verdict(self):
        '''
        Will return -1, 0, or 1 depending on whether it's a lose, draw, 
        or win for the current player.
        '''
        board = self.board
        for i in range(len(board)):
            for j in range(len(board[0])-3):
                if board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3] != 0:
                    return board[i][j]
        for i in range(len(board)-3):
            for j in range(len(board[0])):
                if board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j] != 0:
                    return board[i][j]
        for i in range(3, len(board)):
            for j in range(len(board[0])-3):
                if board[i][j] == board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] != 0:
                    return board[i][j]
        for i in range(len(board)-3):
            for j in range(len(board[0])-3):
                if board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] != 0:
                    return 3 - 2*board[i][j]
        return None

    def get_starting_eval(self):
        if self.turn == 'X':
            return -np.inf
        elif self.turn == 'O':
            return np.inf

    # We can represent a move using i in 1 .. n, where n is the board width, 
    # which represents dropping a token in the i'th position.
    # We represent undoing that same move with -i.
    def get_next_states(self) -> dict:
        moves = np.where(np.array(self.board[0]) == 0)[0] + 1
        return {i: Connect4GameState.play_move(self, i) for i in moves}

    # I actually don't think we'll need this for the current implementation.
    # def get_prev_states(gs: Connect4GameState) -> dict:
    #    pass

    @staticmethod
    def play_move(gs, move):
        board = [row[:] for row in gs.board]

        for i in range(len(board)-1, -1, -1):
            if board[i][move-1] == 0:
                if gs.turn == 'X':
                    board[i][move-1] = 1 
                else:
                    board[i][move-1] = 2
                break

        return Connect4GameState(board, 'X' if gs.turn == 'O' else 'O')
        
# game states is a flat dict of game nodes we use for global memoization.
ttable = {}

class Connect4GameNode:
    def __init__(self, parent, pos, val, children):
        self.parent = parent if parent else None
        self.pos = pos if pos else None
        self.serialized = Connect4Serializer.serialize(self.pos)
        self.val = val if val is not None else self.pos.get_starting_eval()
        self.children = children if children else {}
        self.solved_depth = 0

    def to_dict(self):
        return {
            "val": int(self.val),
            **{"move" + str(int(k)): v.to_dict() for k,v in self.children.items()},
        }

def alphabeta(gs: Connect4GameState, ttable=ttable, depth=5, alpha=-1, beta=1):
    serialized = Connect4Serializer.serialize(gs)
    if serialized not in ttable:
        ttable[serialized] = Connect4GameNode(None, gs, None, gs.get_next_states())
    node = ttable[serialized]

    v = node.pos.verdict()
    if depth == 0 or v is not None:
        node.val = v or 0
        return v or 0
    if node.pos.turn == 'X':
        value = -float('inf')
        for child in random.sample(list(node.children.values()), len(node.children)):
            value = max(value, alphabeta(child, depth=depth-1, alpha=alpha, beta=beta))
            if value >= beta:
                break
            alpha = max(alpha, value)
        node.val = value
        #print(Connect4Serializer.display(serialized), node.val)
        return value
    else:
        value = float('inf')
        for child in random.sample(list(node.children.values()), len(node.children)):
            value = min(value, alphabeta(child, depth=depth-1, alpha=alpha, beta=beta))
            if value <= alpha: 
                break
            beta = min(beta, value)
        node.val = value
        #print(Connect4Serializer.display(serialized), node.val)
        return value

def print_board(board_state, s=False, colnums=False):
    if s:
        res = []
        for row in board_state:
            row_repr = "|"
            for cell in row:
                if cell == 0:
                    row_repr += ' |'
                elif cell == 1:
                    row_repr += 'X|'
                elif cell == 2:
                    row_repr += 'O|'
            res.append(row_repr)
        return res
    
    print('+' + len(board_state[0]) * '-+')
    for row in board_state:
        row_repr = "|"
        for cell in row:
            if cell == 0:
                row_repr += ' |'
            elif cell == 1:
                row_repr += 'X|'
            elif cell == 2:
                row_repr += 'O|'
        print(row_repr)
    print('+' + len(board_state[0]) * '-+')
    if colnums:
        print(' ' + ' '.join([str(n+1) for n in range(len(board_state[0]))])  + ' ')

def input_move(board):
    while True:
        s = input('Select column and press enter: ')
        try:
            n = int(s)
            if not 1 <= n <= 7:
                continue
            else:
                break
        except Exception as e:
            print(e)

    return n

def main():
    print("===== CONNECT 4 ========")
    print("Your turn:\n")

    board = [[0] * 7 for _ in range(6)]
    turn = 'X'
    game_state = Connect4GameState(board, turn)
    # depth = 1

    while True:
        print_board(game_state.board)
        states = game_state.get_next_states()
        while True:
            try: 
                s = int(input("Enter your move:"))
                if s not in states:
                    continue
                else:
                    break
            except Exception as e:
                print(e)
        game_state = states[s]
        print_board(game_state.board)
        print(Connect4Serializer.display(Connect4Serializer.serialize(game_state)))
        if game_state.verdict() is not None:
            print(game_state.verdict())

        states = game_state.get_next_states()
        best_move = min(random.sample(sorted(states), len(states)), key=lambda m: alphabeta(states[m], depth=4))
        game_state = states[best_move]
        if game_state.verdict() is not None:
            print(game_state.verdict())

if __name__ == "__main__":

    
    board = [[0] * 7 for _ in range(6)]
    board[5][3] = board[4][3] = board[3][3] = 1
    board[5][0] = board[4][0] = 2
    turn = 'O'
    game_state = Connect4GameState(board, turn)


    print(alphabeta(game_state, depth=5))


    main()



# def valuation(gs: Connect4GameState, memo_table=valuations, max_depth=float('inf')):
#     serialized = Connect4Serializer.serialize(gs)
#     if serialized not in valuations:
#         valuations[serialized] = Connect4GameNode(None, gs, None, gs.get_next_states())
# 
#     gamenode = valuations[serialized]
# 
#     queue = [gamenode]
# 
#     while len(queue) > 0 and game_node.solved_depth < max_depth:
#         new_queue = []
# 
#         for node in queue:
#             v = node.pos.verdict()
#             if v is notNone:
#                 node.val = v
#                 cur = node
#                 while cur.parent is not None:
#                     cur.parent.val = (max if cur.parent.pos.turn == 'X' else min)(
#                             cur.parent.val, cur.val)
#                     cur = cur.parent
#             else:
#                 new_queue += [node.children[move] for move in self.children]
#                 
#         queue = new_queue
#         # game_node.solved_depth += 1
