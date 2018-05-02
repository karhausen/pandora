# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:51:11 2018

@author: KARHAUSE
"""

import numpy as np
import ElmoNet as en
import matplotlib.pyplot as plt

np.random.seed(1)

#--------------------------------------------

def plot(data):
    plt.plot(data)
    plt.show()

#--------------------------------------------

# number of input and output nodes
input_nodes = 784
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = en.CneuralNetwork(learning_rate)
n.defineInputSize( input_nodes )
n.addLayer( 50, n.sigmoid)
#n.addLayer( 25, n.sigmoid)
#n.addLayer( 8, n.sigmoid)
n.addLayer( output_nodes, n.sigmoid)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
# epochs is the number of times the training data set is used for training

epochs = 200

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


# load the mnist test data CSV file into a list
test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

#n.query(np.asfarray(test_data_list[4].split(',')[1:]) /255.0 *0.99 +0.01)
