# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:44:03 2022

@author: ASUS
"""

import numpy as np
from Layer import Layer

class Softmax(Layer):
    
    def forward(self, input) :
        
        input_expo = np.exp(input)
        self.output = input_expo / np.sum(input_expo)
        return self.output