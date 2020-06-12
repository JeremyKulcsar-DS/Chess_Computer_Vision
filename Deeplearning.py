# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:52:58 2020

@author: Lucas & Jeremy
"""

from Position_recognition import to_fen, to_bitmap
from Image_to_squares import img_to_squares, img_to_coordinates
import matplotlib.pyplot as plt

'''Import required modules.'''
import cv2                       # For reading images
import numpy as np               # For linear algebra

from timeit import default_timer as timer
import tensorflow as tf
import random


from zipfile import ZipFile
import os

#zip = ZipFile('chess-positions.zip')
#print(zip.namelist())


X_coordinates = []
X_aux = []
X_empty = []    #Used to determine if square is empty
Y_empty = []
Y_coordinates = []
X_piece = []    #Used when square is not empty to determine piece on it
Y_piece = []
start = timer()
i=0
j=0

size = 200
'''Here we used data from https://www.kaggle.com/koryakinp/chess-positions to train our models to recognize chess pieces'''
'''Please note that this data is not provided due to its size'''
for filename in os.listdir('positions\\train'):
    if i==10000:
        break
    fen = (filename.replace('-', '/')).split('.')[0]
    file = 'positions\\train\\'+filename
    img = cv2.imread(file)
    img = cv2.resize(img, dsize=(size,size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    squares = img_to_squares(file)
    try:
        T = to_bitmap(fen).reshape(64)
        for k in range(len(squares)):
            if T[k] > 0:
                X_piece.append(squares[k])
                Y_piece.append(T[k]-1)
                X_empty.append(squares[k])
                Y_empty.append(1)
            else:
                X_empty.append(squares[k])
                Y_empty.append(0)
    except:
        #We couldnt recognize chessboard on the picture
        j+=1
        pass
    i+=1
    print(i)

        
X_empty = np.asarray(X_empty)
X_empty = X_empty.reshape((X_empty.shape[0],X_empty.shape[1],X_empty.shape[2],1))
X_empty = X_empty.astype(float)
Y_empty = np.asarray(Y_empty) 
Y_empty = Y_empty.astype(int)


X_piece = np.asarray(X_piece)
X_piece = X_piece.reshape((X_piece.shape[0],X_piece.shape[1],X_piece.shape[2],1))
X_piece = X_piece.astype(float)
Y_piece = np.asarray(Y_piece)
Y_piece = Y_piece.astype(int) 


    
print(timer()-start)
        

'''We have one model to determine wether a square is empty and another model to determine the piece type on not empty squares
We achieved a precision of 100% on train set and 99.99% on test set'''     

from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential, Model

model_empty = Sequential()
model_empty.add(Input(shape=(32,32,1)))
model_empty.add(Conv2D(filters=5, kernel_size=(3,3), activation='relu'))
model_empty.add(Conv2D(filters=5, kernel_size=(3,3), activation='relu'))
model_empty.add(Conv2D(filters=5, kernel_size=(3,3), activation='relu'))
model_empty.add(MaxPooling2D((2,2)))
model_empty.add(Flatten())
model_empty.add(Dense(128, activation='relu'))
model_empty.add(Dense(2, activation='softmax'))

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model_empty.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model_piece = Sequential()
model_piece.add(Input(shape=(32,32,1)))
model_piece.add(Conv2D(filters=5, kernel_size=(3,3), activation='relu'))
model_piece.add(Conv2D(filters=5, kernel_size=(3,3), activation='relu'))
model_piece.add(Conv2D(filters=5, kernel_size=(3,3), activation='relu'))
model_piece.add(MaxPooling2D((2,2)))
model_piece.add(Flatten())
model_piece.add(Dense(256, activation='relu'))
model_piece.add(Dense(12, activation='softmax'))

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model_piece.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])






