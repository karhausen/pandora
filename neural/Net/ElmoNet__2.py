# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 08:02:00 2018

@author: KARHAUSE
"""

import numpy as np
from ..Layer.FCLayer import CFCLayer as layer
#from ..functions import activation_functions as af

# neural network class definition
class CneuralNetwork:
    # initialise the neural network
    def __init__(self, learningrate=0.01, name=''):
        self.name = name
        # define list for layers
        self.layers = []
        # learning rate
        self.lr = learningrate
        pass
    
    def defineInputSize(self, input_size):
        self.input_size = input_size
        pass
    
    def addLayer(self, input_dim, output_dim, activation=None, learning_rate=0.3):
        if activation == None:
            print("No activation function given")
        else:
            depth = len(self.layers)
            if depth > 0:
                 new_layer = layer(input_dim, output_dim, activation, learning_rate)
            else:
                print("first Layer")
                new_layer = layer(input_dim, output_dim, activation, learning_rate)
            self.layers.append(new_layer)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        x = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        layer_count = len(self.layers)
        for layer_index in range(layer_count):
            #print("FW_Layer: {} with x: {}".format(layer_index, x))
            x = self.layers[layer_index].forward(x)
        y = x
        #print("output: {}".format(y) )
        # output layer error is the (target - actual)
        delta = targets -y
        for i in range(layer_count, 0, -1):
            #print("BW_Layer: {} with delta: {}".format(i-1, delta))
            delta = self.layers[i-1].backward(delta)
        pass
 
    # query the neural network
    def query(self, inputs_list):
        x = np.array(inputs_list, ndmin=2).T
        layer_count = len(self.layers)
        for layer_index in range(layer_count):
            #print("Q_Layer: {}".format(layer_index))
            x = self.layers[layer_index].forward(x)
        return x
    
    # backQuery - just for fun
    def backQuery(self, targets_list):
        layer_outputs = np.array(targets_list, ndmin=2).T
        layer_count = len(self.layers)
        for layer_index in range(layer_count-1, 0, -1):
            #print(layer_index)
            w = self.weights[layer_index-1]
            activation_function = self.activation_functions[layer_index-1]
            # calculate the signal into the final output layer
            layer_inputs = activation_function(layer_outputs, True)
            # calculate the signal out of the hidden layer
            layer_outputs = np.dot(w.T, layer_inputs)
            # scale them back to 0.01 to .99
            layer_outputs -= np.min(layer_outputs)
            layer_outputs /= np.max(layer_outputs)
            layer_outputs *= 0.98
            layer_outputs += 0.01
            print(layer_outputs)
        inputs = layer_outputs
        
        return inputs
            

        

 