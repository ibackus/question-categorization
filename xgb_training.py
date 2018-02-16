#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script trains an optimized models on the coarse and fine categories of the
question classification problem.

The model here is a simple bag of words to extract features with a extreme
gradient boosting (xgboost) boosted trees for classification.  A randomized
search on hyperparameters is performed to determine the best hyperparameters,
using a 5-fold cross validation and the matthews correlation coefficient to
score models parameters.

Created on Wed Dec  6 10:57:04 2017

@author: ibackus
"""
# General imports
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
import pickle

# ML imports
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.ensemble # includes ExtraTrees and RandomForest
import sklearn.linear_model
import sklearn.model_selection # Cross validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
mcc_scorer = make_scorer(matthews_corrcoef)
import xgboost

# Project imports
import utils

# setup parallelization
n_clf_jobs = 2
n_search_jobs = int(cpu_count()/n_clf_jobs + 0.5)

# ---------------------------------------------
# Initialize
# ---------------------------------------------
# Load data
train_filename = 'data/train_5500.label'
test_filename = 'data/TREC_10.label'
train = utils.load(train_filename)
test = utils.load(test_filename)

# Set-up the classifier pipeline
pipe = Pipeline([
    ('features', utils.xgb_transformer(CountVectorizer, ngram_range=(1,2))),
    ('classifier', xgboost.XGBClassifier(nthread=n_clf_jobs)),
])

# -------------------------------------------------
# Find the best classifier hyperparameters
# -------------------------------------------------
verbosity = 1

# Set-up the search grid
xgboost_grid = {
    'learning_rate':    utils.gridspace(0.01, 0.3, 100, log=True, integer=False),
    'gamma':            utils.gridspace(0, 1, 100, integer=False),
    'max_depth':        np.arange(3, 101), 
    'max_delta_step':   utils.gridspace(0, 0.1, 100),
    'n_estimators':     utils.gridspace(50, 1000, 200, log=True, integer=True),
    'subsample':        utils.gridspace(0.5, 0.9,100, integer=False),
    'colsample_bytree': utils.gridspace(0.5, 0.9, 100, integer=False)
}
param_grid = {}
for key, val in xgboost_grid.items():
    newkey = 'classifier__' + key
    param_grid[newkey] = val

# Perform search for both categories
labels = ['coarse_category', 'fine_category']
names = ['coarse', 'fine']
for label, name in zip(labels, names):
    print('Performing hyperparameter CV search')
    y_train = train[label]
    y_test = test[label]
    cv = sklearn.model_selection.RandomizedSearchCV(pipe, param_grid, 
        n_iter=50, n_jobs=n_search_jobs, verbose=2, cv=5, scoring=mcc_scorer,
        refit=True)
    cv.fit(train.question, y_train)
    results0 = pd.DataFrame(cv.cv_results_)
    results0.to_csv('results/xgb-{}-grid-search-results.csv'.format(name))
    
    # And, finally, let's see the best params!
    utils.write_fit_params(cv.best_params_, \
                           'results/xgb-{}-model-params.json'.format(name))
    # Save the trained model
    classifier = cv.best_estimator_
    pickle.dump(classifier, open('results/xgb-{}-classifier.p'.format(name), 'wb'))

