# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:51:11 2018

@author: KARHAUSE
"""

import numpy as np
import neural.Net.ElmoNet as en
import neural.functions.activation_functions as af
import matplotlib.pyplot as plt

np.random.seed(1)

#--------------------------------------------

def plot(data):
    plt.plot(data)
    plt.show()

#--------------------------------------------

# Eingangsmatrix
inputs_list = np.array([ [0,0], [0,1], [1,0], [1,1] ])

targets_list = np.array([ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ])


# learning rate
learning_rate = 0.3
#epochs to train
epochs = 50

# create instance of neural network
n = en.CneuralNetwork(learning_rate, name='myNet')

n.defineInputSize( 2 )
#n.addLayer( 128, af.sigmoid, learning_rate)
n.addLayer( 8, af.sigmoid, learning_rate)
n.addLayer( 4, af.sigmoid, learning_rate)

n.train(inputs_list, targets_list, epochs)

print("results after learning:")
for inputs in inputs_list:
    y = n.query(inputs/ 1.0 * 0.98 + 0.01)
    print("{}: {}".format(inputs, np.argmax(y)))
    
#plot(n.training_error)