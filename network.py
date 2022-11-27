# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:12:47 2022

@author: ASUS
"""

def predict(network, input):
    
    for layer in network:
        output = layer.forward(input)
        input = output
        
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):

     for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)