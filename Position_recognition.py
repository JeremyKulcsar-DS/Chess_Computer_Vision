# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:56:30 2020

@author: Lucas & Jeremy
"""


import chess


"""
0 rien
blanc
1 pion P
2 cavalier N
3 fou B
4 tour R
5 queen Q
6 king K
noir
7 pion p
8 cavalier n
9 fou b
10 tour r
11 queen q
12 king k


FEN start
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR

Bitmap start
10,8,9,11,12,9,8,10
7,7,7,7,7,7,7,7
0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0
1,1,1,1,1,1,1,1
4,2,3,5,6,3,2,4

FEN 1 e4
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR

Bitmap 1 e4
10,8,9,11,12,9,8,10
7,7,7,7,7,7,7,7
0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0
0,0,0,0,1,0,0,0
1,1,1,1,0,1,1,1
4,2,3,5,6,3,2,4
"""

import numpy as np

def to_letter(x):
    if x == 1:
        return "P"
    elif x == 2:
        return "N"
    elif x == 3:
        return "B"
    elif x == 4:
        return "R"
    elif x == 5:
        return "Q"
    elif x == 6:
        return "K"
    elif x == 7:
        return "p"
    elif x == 8:
        return "n"
    elif x == 9:
        return "b"
    elif x == 10:
        return "r"
    elif x == 11:
        return "q"
    elif x == 12:
        return "k"
    return ''

def to_fen(bitmap):
    fen = ""
    for k in reversed(range(len(bitmap[0]))):
        count = 0
        for i in range(len(bitmap[0])):
            if to_letter(bitmap[k][i]) == '' and i != len(bitmap[0]) - 1:
                count = count + 1
            elif to_letter(bitmap[k][i]) == '' and i == len(bitmap[0]) - 1:
                count = count + 1
                fen += str(count)
                count = 0
            elif count > 0:
                fen += str(count)
                fen += to_letter(bitmap[k][i])
                count = 0
            else:
                fen += to_letter(bitmap[k][i])
        if k != 0 :
            fen += "/"
    return fen


def to_number(piece):
    number = piece.piece_type
    if piece.color:
        return number
    else:
        return number+6
    

def to_bitmap(fen):
    board = chess.Board(fen)
    O = np.zeros((8,8))
    for i in range(64):
        try:
            number = to_number(board.piece_at(i))
            x,y = i%8, i//8
            O[y][x] = number
        except:
            pass
    return O
        
    