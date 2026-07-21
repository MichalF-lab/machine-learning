import random
import copy
import math

def board(n):
    return [[0 for _ in range(n)] for _ in range(n)]

def possibility_move(board):
    n = len(board)
    moves = []
    # Dodaj ruchy do górnego wiersza
    for i in range(n):
        if board[0][i] == 0:
            moves.append((0, i))
    # Dodaj ruchy do reszty kolumn
    for i in range(1, n):
        for j in range(n):
            if board[i-1][j] in (1, -1) and board[i][j] == 0:
                moves.append((i, j))
    return moves

def check_win_11(tablica):
    """
    Sprawdza, czy gracz 1 (wartość 1) ma cztery symbole w linii.
    Przyjmuje tablicę 7x7 z wartościami 1, -1 i 0.
    """
    n = len(tablica)          # Rozmiar planszy (7)
    player = 1
    for i in range(n):
        for j in range(n):
            if tablica[i][j] == player:
                # Sprawdzenie poziomego ułożenia (w prawo)
                if j <= n-4 and all(tablica[i][j+k] == player for k in range(4)):
                    return True
                # Sprawdzenie pionowego ułożenia (w dół)
                if i <= n-4 and all(tablica[i+k][j] == player for k in range(4)):
                    return True
                # Sprawdzenie ukośnego ułożenia w dół-prawo
                if i <= n-4 and j <= n-4 and all(tablica[i+k][j+k] == player for k in range(4)):
                    return True
                # Sprawdzenie ukośnego ułożenia w górę-prawo
                if i >= 3 and j <= n-4 and all(tablica[i-k][j+k] == player for k in range(4)):
                    return True
    return False

def check_win_12(tablica):
    """
    Sprawdza, czy gracz 2 (wartość -1) ma cztery symbole w linii.
    Działa analogicznie do funkcji dla gracza 1, ale dla wartości -1.
    """
    n = len(tablica)          # Rozmiar planszy (7)
    player = -1
    for i in range(n):
        for j in range(n):
            if tablica[i][j] == player:
                # Sprawdzenie poziomego ułożenia (w prawo)
                if j <= n-4 and all(tablica[i][j+k] == player for k in range(4)):
                    return True
                # Sprawdzenie pionowego ułożenia (w dół)
                if i <= n-4 and all(tablica[i+k][j] == player for k in range(4)):
                    return True
                # Sprawdzenie ukośnego ułożenia w dół-prawo
                if i <= n-4 and j <= n-4 and all(tablica[i+k][j+k] == player for k in range(4)):
                    return True
                # Sprawdzenie ukośnego ułożenia w górę-prawo
                if i >= 3 and j <= n-4 and all(tablica[i-k][j+k] == player for k in range(4)):
                    return True
    return False

def full_board(board):
    for row in board:
        if 0 in row:
            return False
    return True

def play_random_game(board, who_move = 1):
    while True:
        #print(board)
        if check_win_11(board):
            #print("Player 1 wins")
            return 1
        if check_win_12(board):
            #print("Player 2 wins")
            return -1
        if full_board(board):
            return 0

        moves = possibility_move(board)
        move = random.choice(moves)
        board[move[0]][move[1]] = who_move

        who_move *= -1

        return play_random_game(board, who_move)

def move(board, move, who_move):
    new_board = copy.deepcopy(board)
    new_board[move[0]][move[1]] = who_move
    return new_board

def get_id(board, who_move):
    return (tuple(tuple(row) for row in board), who_move)

def main_move(board, who_move):
    tree = {}  #{stan: [wygrane, wizyty]}

    for _ in range(500):
        path = []
        curr_board = copy.deepcopy(board)
        curr_who_move = who_move
        
        
        while True:
            key = get_id(curr_board, curr_who_move)
            moves = possibility_move(curr_board)
            if check_win_11(curr_board) or check_win_12(curr_board) or not moves:
                break
                
            untried = [m for m in moves if get_id(move(curr_board, m, curr_who_move), curr_who_move*-1) not in tree]
            if untried:
                m = random.choice(untried)
                path.append(get_id(curr_board, curr_who_move))
                curr_board = move(curr_board, m, curr_who_move)
                curr_who_move *= -1
                break

            else:
                best_move = None
                best_ucb = -500
                parent_v = tree[key][1]
                
                for m in moves:
                    child_key = get_id(move(curr_board, m, curr_who_move), curr_who_move*-1)
                    w, v = tree[child_key]
                    win_rate = (w / v) if curr_who_move == 1 else (-w / v)
                    ucb = win_rate + 1.41 * math.sqrt(math.log(parent_v) / v)
                    if ucb > best_ucb:
                        best_ucb, best_move = ucb, m
                
                path.append(key)
                curr_board = move(curr_board, best_move, curr_who_move)
                curr_who_move *= -1

        
        res = play_random_game(copy.deepcopy(curr_board), curr_who_move)


        final_key = get_id(curr_board, curr_who_move)
        if final_key not in tree: tree[final_key] = [0, 0]
        tree[final_key][0] += res
        tree[final_key][1] += 1
        for temp_key in path:
            if temp_key not in tree: tree[temp_key] = [0, 0]
            tree[temp_key][0] += res
            tree[temp_key][1] += 1

    moves = possibility_move(board)
    best_move = max(moves, key=lambda m: tree.get(get_id(move(board, m, who_move), who_move*-1), [0, 0])[1])
    return best_move


n = 7
a = board(n)
who_move = 1
for i in range(n*n):
    ext = main_move(a, who_move)
    print(f"Move {i+1}: Player {who_move} moves to {ext}")
    a = move(a, ext, who_move)
    if check_win_11(a):
        print("Player 1 wins!")
        break
    if check_win_12(a):
        print("Player 2 wins!")
        break
    who_move *= -1
for row in a:
    print('[' + ', '.join(f'{x:2}' for x in row) + ']')