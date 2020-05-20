# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:07:18 2019

@author: reblivi
"""

class Problem(object):
    
    def __init__(self, data, label, estimator, cv, **kwargs):
        
        self.data = data
        self.label = label
        self.estimator = estimator
        self.cv = cv
        self.extra = kwargs if kwargs else None
        self.n_rows, self.n_cols = self.data.shape
            
    def _verify_kwargs(self):
        raise NotImplementedError