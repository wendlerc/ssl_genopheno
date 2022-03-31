#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:08:11 2022

@author: chrisw
"""

import numpy as np

class UniformOptimizations:
    def __init__(self, options):
        if type(options) == int:
            self.options = np.arange(options)
        else:
            self.options = options
        self.n = len(self.options)
        
    def randKPerm(self, k):
        return self.options[np.random.permutation(self.n)[:k]]
    
    def randPerm(self):
        k = self.n
        if self.n > 15:
            qkinv = np.e
        else:
            qkinv = np.sum([1/np.math.factorial(m) for m in range(self.n+1)])
        
        while k > 0:
            if np.random.rand() < 1/qkinv:
                break
            else:
                qkinv = (self.n - k + 1)*(qkinv - 1)
                k -= 1
        return self.randKPerm(k)
    
    def __call__(self):
        return self.randPerm()