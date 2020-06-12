# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:40:33 2020

@author: Lucas & Jeremy
"""


from Position_recognition import to_fen, to_bitmap
from Image_to_squares import img_to_squares, img_to_coordinates
import chess
import chess.engine

'''Import required modules.'''
import cv2                       # For reading images
import numpy as np               # For linear algebra
import pyautogui 

import time
from timeit import default_timer as timer
import tensorflow as tf
import random


from zipfile import ZipFile
import os
import sys

from tensorflow.keras.models import load_model
model_empty = load_model('model_empty.h5')
model_piece = load_model('model_piece.h5')

def play(board, engine):
    result = engine.play(board, chess.engine.Limit(time=0.1))
    return result.move

def move_from_boards(chessboard, newboard):
    global NoError
    new_fen = newboard.board_fen()
    for move in chessboard.legal_moves:
        chessboard.push(move)
        if chessboard.board_fen() == new_fen:
            chessboard.pop()
            return move
        else:
            chessboard.pop()
    print('Could not find move')
    print('Restarting AI...')
    NoError = False
    return 

def squares_to_bitmap(squares):
    squares = squares.reshape((squares.shape[0],squares.shape[1],squares.shape[2],1))
    squares = squares.astype(float)
    empty = np.argmax(model_empty.predict(squares), axis=1)
    pieces = np.argmax(model_piece.predict(squares), axis=1)
    bitmap = np.zeros(64)
    for i, e in enumerate(empty):
        if e==1:
            bitmap[i] = pieces[i]+1
    bitmap = bitmap.reshape((8,8))
    return bitmap

    


if __name__ == "__main__": 
    j=1
    
    white_bitmap = to_bitmap('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR')
    black_bitmap = to_bitmap('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'[::-1])
    
    #engine = chess.engine.SimpleEngine.popen_uci('engines\Winter_05\Winter_05_Old.exe')
    #engine = chess.engine.SimpleEngine.popen_uci('engines\stockfish\stockfish.exe')
    engine = chess.engine.SimpleEngine.popen_uci('engines\\arasan\\arasan64')
    #engine = chess.engine.SimpleEngine.popen_uci('engines\gaia\gaia64.exe')
    

    while True:
        mousex, mousey = pyautogui.position()
        if mousex < 3 and mousey < 3:   #PLACE MOUSE IN TOP LEFT CORNER TO TERMINATE SCRIPT
            print('-----------------------')
            print('TERMINATING SCRIPT ON USER DEMAND')
            break
        myturn = False
        iswhite = False
        fen = ''
        NoError = True
        time.sleep(0.5)
        
        while NoError:
            try:
                time.sleep(0.2)
                print('-----------'+str(j)+'-----------')
                print('Screening chessboard...')
                screenshot = np.array(pyautogui.screenshot())
                squares = img_to_squares(screenshot)
                
                if squares.size == 0:
                    print('Chessboard not found')
                    mousex, mousey = pyautogui.position()
                    if mousex < 3 and mousey < 3:   #PLACE MOUSE IN TOP LEFT CORNER TO TERMINATE SCRIPT
                        break
                    
                else:
                    print('Chessboard found')
                    bitmap = squares_to_bitmap(squares)
                    
                    # A new game is beginning
                    if (bitmap == white_bitmap).all():
                        print('Beginning of game as white')
                        chessboard = chess.Board()
                        iswhite = True
                        myturn = True
                    elif (bitmap == black_bitmap).all():
                        print('Beginning of game as black')
                        chessboard = chess.Board()
                        print('Waiting for opponent to play...') 
                        iswhite = False
                        myturn = False
                        continue
                    
                    else:
                        if fen == '':  #Catching game in the middle of it
                            new_fen = to_fen(bitmap)
                            chessboard = chess.Board(new_fen)
                            white_king_row = chessboard.king(True)//8
                            black_king_row = chessboard.king(False)//8
                            if white_king_row <= black_king_row:    #We are likely white
                                print('Catching game as White')
                                iswhite = True
                            else:                                   #We are likely black
                                print('Catching game as Black')
                                iswhite = False
                                new_fen = to_fen(np.flip(bitmap, axis=[0,1]))
                                chessboard = chess.Board(new_fen)
                                chessboard.turn = False
                            fen = new_fen
                            myturn = True  #We assume it is our turn to play
                
                    if iswhite:
                        new_fen = to_fen(bitmap)
                    else:
                        new_fen = to_fen(np.flip(bitmap, axis=[0,1]))
                    
                    if not myturn:
                        if new_fen == fen:
                            print('Waiting for opponent to play...') 
                        else:
                            time.sleep(0.7)
                            screenshot = np.array(pyautogui.screenshot())  #Taking another screenshot prevents from screenshoting while a piece is moving
                            squares = img_to_squares(screenshot)
                            bitmap = squares_to_bitmap(squares)
                            if iswhite:
                                new_fen = to_fen(bitmap)
                            else:
                                new_fen = to_fen(np.flip(bitmap, axis=[0,1]))
                    
                            if new_fen == fen:
                                print('Waiting for opponent to play...') 
                            else:
                                newboard = chess.Board(new_fen)
                                move = move_from_boards(chessboard, newboard)
                                if not NoError:
                                    break
                                chessboard.push(move)
                                fen = new_fen
                                myturn = True
                                
                    if myturn:
                        print('Calculating best move...')
                        best_move = play(chessboard, engine)
                        
                        if iswhite:
                            coordinates = img_to_coordinates(screenshot)
                        else:
                            coordinates = np.flip(img_to_coordinates(screenshot), axis=0)
                        
                        if coordinates.size == 0:
                            print('Could not find chessboard anymore...')
                        else:
                            chessboard.push(best_move)
                            fen = chessboard.board_fen()
                            from_square = coordinates[best_move.from_square]
                            to_square = coordinates[best_move.to_square]
                            time.sleep(random.uniform(0.3,0.5))
                            print('Playing move %s' %best_move)
                            pyautogui.click(*from_square)
                            time.sleep(random.uniform(0.3,0.5))
                            pyautogui.click(*to_square)
                            
                        
                    
                myturn = False
                j+=1
            except:
                print('AI encountered unknown error')
                print('Restarting AI...')
                NoError = False
        
        
    sys.exit()           
                
            
            
        
        
        
    
    
