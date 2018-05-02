# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:06:38 2018

@author: KARHAUSE
"""

import numpy as np

def nothing(x, deriv=False):
    return x

def step_function(x, deriv=False):
    if deriv == False:
        return 1 if x >= 0 else 0
    else:
        return 0 if x >= 0 else 1
    
def sigmoid(x, deriv=False):
    if deriv == False:
        return 1/(1 + np.exp(-x) )
    else:
        x_ = sigmoid(x)
        return x_ * (1 - x_)
    
def tanh(x, deriv=False):
    if deriv == False:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2