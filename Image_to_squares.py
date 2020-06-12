# -*- coding: utf-8 -*-
"""
Created on Wed apr  8 17:53:22 2020

@author: Lucas & Jeremy 
"""


import cv2                       
import numpy as np              
import tensorflow as tf
import scipy.signal
size = 32


def img_to_squares(path):
    if type(path) == str:
        img = cv2.imread(path)
    else:
        img = path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    
    thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    sobelx = cv2.Sobel(close, cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(close, cv2.CV_64F,0,1,ksize=3)
    
    Dx_pos = tf.clip_by_value(sobelx, 0., 255., name="dx_positive")
    Dx_neg = tf.clip_by_value(sobelx, -255., 0., name='dx_negative')
    Dy_pos = tf.clip_by_value(sobely, 0., 255., name="dy_positive")
    Dy_neg = tf.clip_by_value(sobely, -255., 0., name='dy_negative')
    
    hough_Dx = tf.reduce_sum(Dx_pos, 0) * tf.reduce_sum(-Dx_neg, 0) / (close.shape[0]*close.shape[0])
    hough_Dy = tf.reduce_sum(Dy_pos, 1) * tf.reduce_sum(-Dy_neg, 1) / (close.shape[1]*close.shape[1])
    hough_Dx_thresh = tf.reduce_max(hough_Dx) * 2 / 5
    hough_Dy_thresh = tf.reduce_max(hough_Dy) * 2 / 5
    
    lines_x, lines_y, is_match = getChessLines(hough_Dx.numpy().flatten(), hough_Dy.numpy().flatten(), hough_Dx_thresh.numpy()*.9, hough_Dy_thresh.numpy()*.9)
    
    if is_match:
        #print("Chessboard found")
        # Possibly check np.std(np.diff(lines_x)) for variance etc. as well/instead
        #print("7 horizontal and vertical lines found, slicing up squares")
        squares = getChessTiles(gray, lines_x, lines_y)
        #print("Tiles generated: (%dx%d)*%d" % (squares.shape[0], squares.shape[1], squares.shape[2]))
        return squares
    else:
        # print ("Couldn't find Chessboard")
        # print("Number of lines not equal to 7")
        return np.array([])


####################################################################################################################################################################


def checkMatch(lineset):
    """Checks whether there exists 7 lines of consistent increasing order in set of lines"""
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    for line in linediff:
        # Within 5 px of the other (allowing for minor image errors)
        if np.abs(line - x) < 5:
            cnt += 1
        else:
            cnt = 0
            x = line
    return cnt == 5

def pruneLines(lineset):
    """Prunes a set of lines to 7 in consistent increasing order (chessboard)"""
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    start_pos = 0
    for i, line in enumerate(linediff):
        # Within 5 px of the other (allowing for minor image errors)
        if np.abs(line - x) < 5:
            cnt += 1
            if cnt == 5:
                end_pos = i+2
                return lineset[start_pos:end_pos]
        else:
            cnt = 0
            x = line
            #print (i, x)
            start_pos = i
    return lineset

def skeletonize_1d(arr):
    """return skeletonized 1d array (thin to single value, favor to the right)"""
    _arr = arr.copy() # create a copy of array to modify without destroying original
    # Go forwards
    for i in range(_arr.size-1):
        # Will right-shift if they are the same
        if arr[i] <= _arr[i+1]:
            _arr[i] = 0
    
    # Go reverse
    for i in np.arange(_arr.size-1, 0,-1):
        if _arr[i-1] > _arr[i]:
            _arr[i] = 0
    return _arr

def getChessLines(hdx, hdy, hdx_thresh, hdy_thresh):
    """Returns pixel indices for the 7 internal chess lines in x and y axes"""
    # Blur
    gausswin = scipy.signal.gaussian(21,4)
    gausswin /= np.sum(gausswin)

    # Blur where there is a strong horizontal or vertical line (binarize)
    blur_x = np.convolve(hdx > hdx_thresh, gausswin, mode='same')
    blur_y = np.convolve(hdy > hdy_thresh, gausswin, mode='same')


    skel_x = skeletonize_1d(blur_x)
    skel_y = skeletonize_1d(blur_y)

    # Find points on skeletonized arrays (where returns 1-length tuple)
    lines_x = np.where(skel_x)[0] # vertical lines
    lines_y = np.where(skel_y)[0] # horizontal lines
    
    # Prune inconsistent lines
    lines_x = pruneLines(lines_x)
    lines_y = pruneLines(lines_y)
    
    is_match = len(lines_x) == 7 and len(lines_y) == 7 and checkMatch(lines_x) and checkMatch(lines_y)
    
    return lines_x, lines_y, is_match

def getChessTiles(a, lines_x, lines_y):
    global size
    """Split up input grayscale array into 64 tiles stacked in a 3D matrix using the chess linesets"""
    # Find average square size, round to a whole pixel for determining edge pieces sizes
    stepx = np.int32(np.round(np.mean(np.diff(lines_x))))
    stepy = np.int32(np.round(np.mean(np.diff(lines_y))))
    
    # Pad edges as needed to fill out chessboard (for images that are partially over-cropped)
#     print stepx, stepy
#     print "x",lines_x[0] - stepx, "->", lines_x[-1] + stepx, a.shape[1]
#     print "y", lines_y[0] - stepy, "->", lines_y[-1] + stepy, a.shape[0]
    padr_x = 0
    padl_x = 0
    padr_y = 0
    padl_y = 0
    
    if lines_x[0] - stepx < 0:
        padl_x = np.abs(lines_x[0] - stepx)
    if lines_x[-1] + stepx > a.shape[1]-1:
        padr_x = np.abs(lines_x[-1] + stepx - a.shape[1])
    if lines_y[0] - stepy < 0:
        padl_y = np.abs(lines_y[0] - stepy)
    if lines_y[-1] + stepx > a.shape[0]-1:
        padr_y = np.abs(lines_y[-1] + stepy - a.shape[0])
    
    # New padded array
#     print "Padded image to", ((padl_y,padr_y),(padl_x,padr_x))
    a2 = np.pad(a, ((padl_y,padr_y),(padl_x,padr_x)), mode='edge')
    
    setsx = np.hstack([lines_x[0]-stepx, lines_x, lines_x[-1]+stepx]) + padl_x
    setsy = np.hstack([lines_y[0]-stepy, lines_y, lines_y[-1]+stepy]) + padl_y
    
    a2 = a2[setsy[0]:setsy[-1], setsx[0]:setsx[-1]]
    setsx -= setsx[0]
    setsy -= setsy[0]
#     display_array(a2, rng=[0,255])    
#     print "X:",setsx
#     print "Y:",setsy
    
    # Matrix to hold images of individual squares (in grayscale)
#     print "Square size: [%g, %g]" % (stepy, stepx)
    squares = np.zeros([64, size, size],dtype=np.uint8)
    
    # For each row
    for i in range(0,8):
        # For each column
        for j in range(0,8):
            # Vertical lines
            x1 = setsx[i]
            x2 = setsx[i+1]
            padr_x = 0
            padl_x = 0
            padr_y = 0
            padl_y = 0

            if (x2-x1) > stepx:
                if i == 7:
                    x1 = x2 - stepx
                else:
                    x2 = x1 + stepx
            elif (x2-x1) < stepx:
                if i == 7:
                    # right side, pad right
                    padr_x = stepx-(x2-x1)
                else:
                    # left side, pad left
                    padl_x = stepx-(x2-x1)
            # Horizontal lines
            y1 = setsy[j]
            y2 = setsy[j+1]

            if (y2-y1) > stepy:
                if j == 7:
                    y1 = y2 - stepy
                else:
                    y2 = y1 + stepy
            elif (y2-y1) < stepy:
                if j == 7:
                    # right side, pad right
                    padr_y = stepy-(y2-y1)
                else:
                    # left side, pad left
                    padl_y = stepy-(y2-y1)
            # slicing a, rows sliced with horizontal lines, cols by vertical lines so reversed
            # Also, change order so its A1,B1...H8 for a white-aligned board
            # Apply padding as defined previously to fit minor pixel offsets
            s = np.pad(a2[y1:y2, x1:x2],((padl_y,padr_y),(padl_x,padr_x)), mode='edge')
            s = cv2.resize(s, dsize=(size,size))
            squares[(7-j)*8+i,:,:] = s
    return squares


##########################################################################################################################
    
def img_to_coordinates(path):
    if type(path) == str:
        img = cv2.imread(path)
    else:
        img = path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    
    thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    sobelx = cv2.Sobel(close, cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(close, cv2.CV_64F,0,1,ksize=3)
    
    Dx_pos = tf.clip_by_value(sobelx, 0., 255., name="dx_positive")
    Dx_neg = tf.clip_by_value(sobelx, -255., 0., name='dx_negative')
    Dy_pos = tf.clip_by_value(sobely, 0., 255., name="dy_positive")
    Dy_neg = tf.clip_by_value(sobely, -255., 0., name='dy_negative')
    
    hough_Dx = tf.reduce_sum(Dx_pos, 0) * tf.reduce_sum(-Dx_neg, 0) / (close.shape[0]*close.shape[0])
    hough_Dy = tf.reduce_sum(Dy_pos, 1) * tf.reduce_sum(-Dy_neg, 1) / (close.shape[1]*close.shape[1])
    hough_Dx_thresh = tf.reduce_max(hough_Dx) * 2 / 5
    hough_Dy_thresh = tf.reduce_max(hough_Dy) * 2 / 5
    
    lines_x, lines_y, is_match = getChessLines(hough_Dx.numpy().flatten(), hough_Dy.numpy().flatten(), hough_Dx_thresh.numpy()*.9, hough_Dy_thresh.numpy()*.9)
    
    if is_match:
        #print("Chessboard found")
        # Possibly check np.std(np.diff(lines_x)) for variance etc. as well/instead
        #print("7 horizontal and vertical lines found, slicing up squares")
        coordinates = getChessTilesCoordinates(gray, lines_x, lines_y)
        #print("Tiles generated: (%dx%d)*%d" % (squares.shape[0], squares.shape[1], squares.shape[2]))
        return coordinates
    else:
        # print ("Couldn't find Chessboard")
        # print("Number of lines not equal to 7")
        return np.array([])





#############################################################################################################################
    
def getChessTilesCoordinates(a, lines_x, lines_y):
    # Find average square size, round to a whole pixel for determining edge pieces sizes
    stepx = np.int32(np.round(np.mean(np.diff(lines_x))))
    stepy = np.int32(np.round(np.mean(np.diff(lines_y))))
    
    # Pad edges as needed to fill out chessboard (for images that are partially over-cropped)
    padr_x = 0
    padl_x = 0
    padr_y = 0
    padl_y = 0
    
    if lines_x[0] - stepx < 0:
        padl_x = np.abs(lines_x[0] - stepx)
    if lines_x[-1] + stepx > a.shape[1]-1:
        padr_x = np.abs(lines_x[-1] + stepx - a.shape[1])
    if lines_y[0] - stepy < 0:
        padl_y = np.abs(lines_y[0] - stepy)
    if lines_y[-1] + stepx > a.shape[0]-1:
        padr_y = np.abs(lines_y[-1] + stepy - a.shape[0])
    
    setsx = np.hstack([lines_x[0]-stepx, lines_x, lines_x[-1]+stepx]) + padl_x
    setsy = np.hstack([lines_y[0]-stepy, lines_y, lines_y[-1]+stepy]) + padl_y

    centers_x = np.zeros(len(setsx)-1)
    centers_y = np.zeros(len(setsy)-1)
    centers = np.zeros((len(setsx)-1, len(setsy)-1,  2))
    
    for i in range(len(setsx)-1):
        centers_x[i] = (setsx[i+1]+setsx[i])/2
    
    for i in range(len(setsy)-1):
        centers_y[i] = (setsy[i+1]+setsy[i])/2

    for i in range(len(setsx)-1):
        for j in range(len(setsy)-1):
            centers[i][j] = (centers_x[i],centers_y[j])
    
    centers_coordinates = np.zeros(((len(setsx)-1)*(len(setsy)-1),  2))
    l = 0
    for i in reversed(range(len(setsy)-1)):
        for j in range(len(setsx)-1):
            centers_coordinates[l] = centers[j][i]
            l = l+1
    
    return centers_coordinates