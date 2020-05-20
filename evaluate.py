# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:12:08 2019

@author: reblivi
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


class SolutionEvaluator(object):
    
    
    def __init__(self, p, size_pop):
        
        self.p = p
        self.cv = p.cv
        self.calls = 0
        self.evolution = []
        self.best = 0.0
        self.size_pop = size_pop
        self.n_cols = self.p.n_cols - 1
        self.totaliter = []
        self.data = self.p.data
        self.label = self.p.label
        self.cvs = None
        self.cv = self.p.cv
        self.estimator = self.p.estimator
        self.extra = self.p.extra
        
    def evaluate(self, pop):
        
        tam = pop.shape[1] - 1
        x_train = self.data
        y_train = self.label
        
        
        self.cvs = self._get_cv(y_train)
        for i in range(0, self.size_pop):
            
            lista = self.drop_features(pop[i, :])
            if(self.p.n_cols - len(lista)) > 1:
                
                if isinstance(x_train, pd.DataFrame):
                    dados_train = x_train.drop(x_train.columns[[lista]], 1)
                    
                    
                if sparse.issparse(x_train):
                    dados_train = x_train[:, lista]
                    
                    
                pred = self.model(dados_train, y_train)    
                
                pop[i ,tam] = pred
                self.totaliter.append(pop[i, :])
                
        return pop


    def drop_features(self, s, threshold=0.6):
        
        listt = []
        for i in range(0, len(s) - 1):
            if s[i] <= threshold:
                listt.append(i)

        return listt
    
    
    
    def model(self, data, y_train):
        
        
        
        module_name = self.estimator.__module__
        if 'xgboost' in module_name:
            result = self._tune_xgb(data, y_train)
        elif 'lightgbm' in module_name:
            result = self._tune_lgb(data, y_train)
            
        else:
            result = self._tune_generic(data, y_train)
            
        return result


    def _tune_xgb(self, x_tr, y_tr):
        
        lista_models = []
        
        x_tr = np.array(x_tr)
        y_tr = np.array(y_tr)


        for train_index, test_index in self.cvs.split(x_tr, y_tr):
            
            
            train_x, valid_x = x_tr[train_index], x_tr[test_index]
            train_y, valid_y = y_tr[train_index], y_tr[test_index]
            
            self.estimator.fit(
                    train_x, train_y,
                    eval_set=[(valid_x, valid_y)],
                    eval_metric=self.extra['eval_metric'],
                    early_stopping_rounds=self.extra['early_stopping_rounds'],
                    verbose=self.extra['verbose'])
            lista_models.append(self.estimator.best_score)
                  
        return np.mean(lista_models)    
                         
    def _tune_lgb(self, x_tr, y_tr):
        
        lista_models = []
        
        x_tr = np.array(x_tr)
        y_tr = np.array(y_tr)

        for train_index, test_index in self.cvs.split(x_tr, y_tr):
            
            train_x, valid_x = x_tr[train_index], x_tr[test_index]
            train_y, valid_y = y_tr[train_index], y_tr[test_index]
            
            self.estimator.fit(
                    train_x, train_y,
                    eval_set=[(valid_x, valid_y)],
                    eval_metric=self.extra['eval_metric'],
                    early_stopping_rounds=self.extra['early_stopping_rounds'],
                    verbose=self.extra['verbose'])
            lista_models.append(self.estimator.best_score_['valid_0'][self.extra['eval_metric']])
            
            
            
        return np.mean(lista_models)                    
                    
            
            
    def _tune_generic(self, x_tr, y_tr):
        
        lista_models = []
        
        x_tr = np.array(x_tr)
        y_tr = np.array(y_tr)
        
        for train_index, test_index in self.cvs.split(x_tr, y_tr):
            
            
            train_x, valid_x = x_tr[train_index], x_tr[test_index]
            train_y, valid_y = y_tr[train_index], y_tr[test_index]
            
            self.estimator.fit(
                    train_x, train_y)
            
            if self.extra['eval_metric'] == 'auc':
                
                pred = self.estimator.predict_proba(valid_x)[:, 1]
                lista_models.append(roc_auc_score(valid_y, pred))
                      
        return np.mean(lista_models)                 
                 
    def _get_cv(self, y_tr):
        
        cv = check_cv(self.cv, y_tr, classifier=is_classifier(self.estimator))
        return cv
                
                    
        
        