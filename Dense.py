# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:41:33 2022

@author: ASUS
"""
from Layer import Layer
import numpy as np

class Dense(Layer):
    
    def __init__(self, input_size, output_size):
        
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size,1)
        
    def forward(self, input):
        self.input = input
        return (np.dot(self.weights, input) + self.biases)
    
    def backward(self, output_gradient, learning_rate):
        
        input_gradient = np.dot(self.weights.T, output_gradient)
        weights_gradient = np.dot(output_gradient, self.input.T)
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        
        
        return input_gradient