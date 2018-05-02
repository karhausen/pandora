# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:16:47 2018

@author: KARHAUSE
"""
import numpy as np
import neural.layers.FCLayer as nl
import neural.functions.activation_functions as af


if __name__ == "__main__":
    np.random.seed(1)
    
    inputs_list     = np.array([0,1])
    targets_list    = np.array([1,0])
    hidden_nodes    = 4
    learning_rate   = 0.3
    
    l1 = nl.CFCLayer(2, hidden_nodes, af.sigmoid, learning_rate)
    l2 = nl.CFCLayer(hidden_nodes,2, af.sigmoid, learning_rate)
    
    x = np.array(inputs_list, ndmin=2).T
    y = np.array(targets_list, ndmin=2).T
    
    for i in range(2):
        l1_out = l1.forward(x)
        l2_out = l2.forward(l1_out)
        
        l2_delta = y - l2_out 
        l1_delta = l2.backward(l2_delta)
        l1.backward(l1_delta)
        
    
        output = l2.forward(l1.forward(x))
        print(output)