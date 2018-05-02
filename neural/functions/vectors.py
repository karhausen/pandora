# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:41:57 2018

@author: KARHAUSE
"""

import numpy as np
import matplotlib.pyplot as plt

def sum_of_squares(v):
    """v_1 * v_1 + v_2 * v_2 .... v_n * v_n """
    return np.dot(v, v)

def magnitude(v):
    return np.sqrt(sum_of_squares(v))

def squared_distance(v,w):
    return sum_of_squares(np.subtract(v, w))

def distance(v, w):
    return np.sqrt(squared_distance(v, w))

def split(u):
    x = []
    y = []
    for item in data:
        x_, y_ = item
        x.append(x_)
        y.append(y_)
    return x, y

def nearest_neighbor(x, u):
    dist_table = []
    for item in u:
        dist_table.append([distance(x, item)])
    i = np.argmin(dist_table)
    return i, dist_table

#------------------------------------------

v = np.array([1,1,1])
w = np.array([1,2,2])

data = [[2.5, 1.0], # a
        [1.5, 3.5],
        [4.0, 5.0],
        [6.5, 2.5],
        [7.0, 6.5],
        [2.5, 7.0],
        [1.5, 2.0],
        [8.0, 4.0],
        [5.5, 7.5], # i
        [4.0, 6.5],
        [6.5, 4.5],
        [1.5, 5.0],
        [5.0, 1.0],
        [4.0, 1.5], # n
        [8.5, 2.0],
        [7.5, 1.5],
        [7.5, 5.0],
        [1.0, 1.0],
        [1.0, 6.5],
        [3.5, 8.0],
        [5.5, 5.0], # u
        [4.0, 4.0],
        [3.0, 3.5],
        [2.5, 4.0],
        [3.0, 5.5],
        [3.0, 2.0]]

labels = ['a', 'b', 'c', 'd', 'e', 'f',
           'g', 'h', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p', 'q', 'r',
           's', 't', 'u', 'v', 'w', 'x',
           'y', 'z ']

# i, nn = nearest_neighbor([5.5, 6.5], data)
# 