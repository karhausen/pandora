# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:57:35 2018

@author: KARHAUSE
"""
import numpy as np

class CBase(object):    
    def __init__(self):
        
        pass
    
    def __repr__(self):
        classname = self.__class__.__name__
        format_str = '{name}(input_dim={input_dim}, '\
                     'output_dim={output_dim}, '\
                     'activation_function={activation}, '\
                     'learning_rate={lr})'
        return format_str.format(name=classname, 
                                 input_dim=self.input_dim, 
                                 output_dim=self.output_dim,
                                 activation=self.activation_function,
                                 lr=self.learning_rate)
        
class CInput(CBase):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.activation_function = None
        self.learning_rate = None
        pass
    
    def forward(self,inputs_list):
        self.input =  inputs_list
        self.output = self.input
        return self.output
    
    def backward(self,output_delta):
        return output_delta
    
class CLayer(CBase):    
    def __init__(self, input_dim, output_dim, activation, learning_rate=0.3, dropout=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.weights = np.random.normal(0.0, pow(input_dim, -0.5), ( output_dim, input_dim))
        self.activation_function = activation
        self.learning_rate = learning_rate

    def forward(self,inputs_list):
        if len(inputs_list.shape) == 1:
            self.input =  np.array(inputs_list, ndmin=2).T
            self.output = self.activation_function(self.input.T.dot(self.weights))
        else:
            self.input = inputs_list
            self.output = self.activation_function(np.dot(self.weights, self.input))
        return self.output
    
    def backward(self,output_delta):
        hidden_error = np.dot(self.weights.T, output_delta)
        self.weights += self.learning_rate * np.dot( (output_delta * self.output * (1.0 - self.output)), np.transpose(self.input))
        return hidden_error
    
  