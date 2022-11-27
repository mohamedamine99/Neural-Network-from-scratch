# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:25:16 2022

@author: ASUS
"""
from Layer import Layer
import numpy as np

class Activation(Layer):
    
    def __init__(self, activation , activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    