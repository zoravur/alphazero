import numpy as np
import random
import collections

class Node:
    def __init__(self, board_state, turn='X'):
        self.pos = board_state
        self.val = None
        self.turn = turn
        self.children = {} # keys are 1-7, representing the possible moves

    def print(self):
        print('===NODE===')
        print(self.val)
        print(self.turn)
        print(self.pos)
        for move in self.children:
            self.children[move].print()

    def to_dict(self):
        # recursively convert node and children to a dictionary
        return {
            **{i: row for i, row in enumerate(print_board(self.pos, s=True))},
            "val": int(self.val),
            "turn": self.turn,
            **{"move" + str(int(k)): v.to_dict() for k,v in self.children.items()},
        }

def maximin(board_state, turn='X'):
    tree = Node(board_state, turn)

    visited_states = collections.defaultdict(lambda: None)
    def compute_val(node):
        hashable_array = tuple(node.pos.flatten())
        if visited_states[hashable_array] is not None:
            return visited_states[hashable_array].val

        v = verdict(node.pos)
        # print(node.pos)
        if v is not None:
            return v

        moves = get_valid_moves(node.pos).tolist()
        for move in random.sample(moves, len(moves)):
            bs = np.copy(node.pos)
            play_move(bs, move, node.turn)
            #if visited_states[tuple(bs.flatten())] is not None:
            #    node.children[move] = visited_states[tuple(bs.flatten())]
            #else:
            node.children[move] = Node(bs, 'O' if node.turn == 'X' else 'X')
            node.children[move].val = compute_val(node.children[move])
            if node.children[move].val == 1 and node.turn == 'X' or node.children[move].val == -1 and node.turn == 'O':
                break

        node.val = np.max([node.children[move].val for move in node.children]) if node.turn == 'X' else np.min([node.children[move].val for move in node.children])

        visited_states[hashable_array] = node
        return visited_states[hashable_array].val

    tree.val = compute_val(tree)
    return tree

def check_valid(board_state, move='X'):
    board_state = np.pad(np.array(board_state), ((3, 3), (3, 3)), 'constant', constant_values=0)    
    #print(board_state)
    offsets = np.array([[(1, -1)], [(1, 0)], [(1, 1)], [(0, 1)]])

    for i in range(3,len(board_state)-3):
        for j in range(3,len(board_state[0])-3):
            #print([i,j])
            posns = (offsets * np.array([[0],[1],[2],[3]]) + [i, j])
            if move=='X':
                if any(np.prod(np.array(list(map(lambda k: list(map(lambda arr: board_state[tuple(arr)] == 1, k)), posns))),-1)):
                    return True
            elif move=='O':
                if any(np.prod(np.array(list(map(lambda k: list(map(lambda arr: board_state[tuple(arr)] == 2, k)), posns))),-1)):
                    return True
            else:
                raise Exception('unknown turn (must be X or O)')
    return False

def verdict(board_state):
    if check_valid(board_state, move='X'):
        return 1
    if check_valid(board_state, move='O'):
        return -1
    if len(get_valid_moves(board_state)) == 0:
        return 0
    return None

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

def new_board(m, n):
    return np.array([[0] * n for _ in range(m)])

def get_valid_moves(board):
    return np.where(np.array(board)[0] == 0)[0]+1

def play_move(board, col, piece):
    for i in range(len(board)-1, -1, -1):
        if board[i][col-1] == 0:
            if piece == 'X':
                board[i][col-1] = 1 
            else:
                board[i][col-1] = 2
            break

def input_move(board, turn='X'):
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

    play_move(board, n, turn)
    return n

def computer_move(board):
    play_move(board, random.choice(get_valid_moves(board)), 'O')

def game_interactive():
    print("====== CONNECT 4 ==========")
    print("Your turn:\n")

    board = new_board(6,7)

    while True:
        print_board(board, colnums=True)
        input_move(board)
        computer_move(board)
        if verdict(board) == 1:
            print_board(board)
            print("YOU WIN!!")
            break
        elif verdict(board) == -1:
            print_board(board)
            print("YOU LOSE!")
            break
        else:
            verdict(board) == 0
            print_board(board)
            print("tie!")

def read_verdict(board):
    if verdict(board) == 1:
        print_board(board)
        print("X WINS!")
        return True
    elif verdict(board) == -1:
        print_board(board)
        print("O WINS!")
        return True
    elif verdict(board) == 0:
        print_board(board)
        print("tie!")
        return True
    return False


def game_interactive_maximin(computer_first=False):
    print("Initializing board")
    board = new_board(3,4)

    print("Computing gametree")
    t = maximin(board)

    print("===== CONNECT 4 ==========")
    print("Your turn:\n")


    if computer_first:
        while True:
            best_move = max(t.children, key=lambda m: t.children[m].val)
            play_move(board, best_move, 'X')
            t = t.children[best_move]
            print(f"Computer played {best_move}")
            if read_verdict(board):
                break

            print_board(board, colnums=True)
            move = input_move(board, 'O')
            t = t.children[move]
            if read_verdict(board):
                break
    
    else:
        while True:
            print_board(board, colnums=True)
            print(get_valid_moves(board))
            move = input_move(board)
            t = t.children[move]
            if read_verdict(board):
                break
            #computer move
            # print(get_valid_moves(board))
            # print(t.children)
            best_move = min(t.children, key=lambda m: t.children[m].val)
            play_move(board, best_move, 'O')
            t = t.children[best_move]
            print(f"Computer played {best_move}")
            if read_verdict(board):
                break

def main():
    import json
    #print(maximin(new_board(1,1)).__dict__)
    #print(maximin(new_board(2,2)).__dict__)
    #print(maximin(new_board(3,3)).__dict__)
    # result = maximin(np.array([
    #         [1,0,0,0],
    #         [2,2,2,0],
    #         [1,1,1,2],
    #         ]))

    # game_interactive_maximin(computer_first=True)

    # print(verdict(np.array([[2,0,0,0],[2,0,1,0],[2,1,1,1]])))

    # print(json.dumps(maximin(np.array([[2,0,0,0],[2,0,1,0],[2,1,1,1]]),turn='O').to_dict(), indent=3))

    # # board = np.array([
    # #         [0,2,0,0],
    # #         [1,2,0,0],
    # #         [1,2,0,1],
    # #         ])
    board = new_board(3, 4)

    result = maximin(board)

    print(json.dumps(result.to_dict(), indent=3))


if __name__ == '__main__':
    main()
