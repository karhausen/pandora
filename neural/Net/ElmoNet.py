# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 08:02:00 2018

@author: KARHAUSE
"""

import numpy as np
from ..layers.Layer import CLayer, CInput
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
        self.training_error = []
        pass
    
    def defineInputSize(self, input_size):
        self.input_size = input_size
        new_layer = CInput(input_size)
        self.layers.append(new_layer)
        pass
    
    def addLayer(self, dim, activation=None, learning_rate=0.3):
        if activation == None:
            print("No activation function given")
        else:
            depth = len(self.layers)
            output_dim = dim
            if depth > 0:
                input_dim = self.layers[depth-1].output_dim
                new_layer = CLayer(input_dim, output_dim, activation, learning_rate)
                self.layers.append(new_layer)
    
    def getWeights(self, layer):
        if (layer > 0) and (layer < len(self.layers)):
            return self.layers[layer].weights

    # train the neural network
    def train(self, inputs_list, targets_list, epochs=1):
        results = []
        for e in range(epochs):
            # go through all records in the training data set
            error = 0
            for i in range(len(inputs_list)):
                # scale and shift the inputs
                inputs = inputs_list[i]#/ 1.0 * 0.98 + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = targets_list[i]#/ 1.0 * 0.98 + 0.01
                #print("{}: {}".format(inputs, targets))        
        
                # convert inputs list to 2d array
                x = np.array(inputs_list, ndmin=2).T
                targets__ = np.array(targets_list, ndmin=2).T
                layer_count = len(self.layers)
                for layer_index in range(layer_count):
                    #print("FW_Layer: {} with x: {}".format(layer_index, x))
                    x = self.layers[layer_index].forward(x)
                # there is no layer more, we've finished our layer chain
                y = x
                # output layer error is the (target - actual)
                delta = targets__ -y
                for layer_index in range(layer_count, 0, -1):
                    #print("BW_Layer: {} with delta: {}".format(i-1, delta))
                    delta = self.layers[layer_index-1].backward(delta)
            test = self.query(inputs)
            error += (test - targets)**2
            #print("{} {}".format(test, error))
            results.append(error[0]/4)                  
        self.training_error = results            
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
        for layer_index in range(layer_count, 1, -1):
            print(layer_index)
            w = self.layers[layer_index-1].weights
            activation_function = self.layers[layer_index-1].activation_function
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
            

        

 