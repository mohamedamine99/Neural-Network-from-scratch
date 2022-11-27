# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:32:44 2022

@author: ASUS
"""

import numpy as np
from Activation import Activation
from Layer import Layer

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
         

class Sigmoid(Activation):
    
    def __init__(self):
        
        def sigmoid(x):
            return (  1 / (1 + np.exp(-x))  )
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1-s)
        
class Softmax(Layer):
    
    def forward(self, input) :
        
        input_expo = np.exp(input)
        self.output = input_expo / np.sum(input_expo)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
         
    