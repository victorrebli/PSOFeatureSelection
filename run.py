from importlib import reload
from scipy.io import arff
#import utils
#reload(utils)
#from ETree import Tree_expression, ParseTree
import copy
#import evaluate
#from evaluate import GPFactory
#reload(evaluate)
import principal
from principal import PSOSelector
reload(principal)
import lightgbm as lgb
import numpy as np
import collections
import pandas as pd
import sys


data = arff.loadarff('bases/sonar.arff')
df_data = pd.DataFrame(data[0])

y_train = df_data.Class
X_train = df_data.drop('Class', axis=1)
y_train = np.where(y_train == b'Rock', 1, 0)

lgb_params = {}
lgb_params['n_jobs'] = 3
lgb_params['is_unbalance'] = True
lgb_params['boosting_type'] = 'gbdt'
lgb_params['objective'] = 'binary'
lgb_params['num_leaves'] = 31
lgb_params['learning_rate'] = 0.01
lgb_params['verbose'] = 1
lgb_params['max_depth'] = 5
lgb_params['min_split_gain'] = 0.002
lgb_params['verbosity'] = -1
lgb_params['n_estimators'] = 10000
lgb_params['random_state'] = 42

estimator = lgb.LGBMClassifier(**lgb_params)

extras = {'eval_metric': 'auc',
          'early_stopping_rounds': 100,
         'verbose': False}


model = PSOSelector(estimator, w=0.7298, c1=1.49618, c2=1.49618,
                    num_particles=30, max_iter=100, max_local_improvement=50,
                    maximize_objective=True, initialization='uniform',
                    fitness_method='type_2', cv = 3)

model.fit(X_train,y_train, **extras)