#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script trains an optimized models on the coarse category of the
question classification problem.

The model here is a simple bag of words to extract features with an 
extra-trees ensemble classifier.

Grid searches on hyperparameters are performed to determine the best 
hyperparameters, using a 5-fold cross validation and the matthews correlation
coefficient to score models parameters.

The grid searches here were used to understand the system and the bag of words
feature extractor.  The series of grid searches informed the following grid
searches.

Created on Wed Dec  6 10:57:04 2017

@author: ibackus
"""
# General imports
from multiprocessing import cpu_count
import pandas as pd
import pickle

# Sckit-learn imports
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.ensemble # includes ExtraTrees and RandomForest
import sklearn.linear_model
import sklearn.model_selection # Cross validation
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
mcc_scorer = make_scorer(matthews_corrcoef)

# Project specific imports
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
    ('features', CountVectorizer()),
    ('classifier', ExtraTreesClassifier(max_depth=75, random_state=88,
                       n_estimators=50, n_jobs=n_clf_jobs)),
])

# -------------------------------------------------
# Find the best classifier hyperparameters
# -------------------------------------------------
verbosity = 1

# Perform a broad hyperparameter grid search
print('Performing broad grid search')
ngram_ranges = [(1,1), (1,2), (1,3)]
stop_words = [None, 'english']
param_grid = [
    {
        'features__ngram_range': ngram_ranges,
        'features__stop_words': stop_words,
        'classifier__max_depth': utils.gridspace(2, 150, n=4, log=True),
        'classifier__n_estimators': utils.gridspace(5, 150, n=4, log=True)
    }
]
cv0 = sklearn.model_selection.GridSearchCV(pipe, param_grid, 
                                          n_jobs=n_search_jobs, cv=5,
                                          refit=True, verbose=verbosity,
                                          scoring=mcc_scorer)
cv0.fit(train.question, train.coarse_category)
results0 = pd.DataFrame(cv0.cv_results_)
results0.to_csv('results/coarse-grid-search-results.csv')
print('Broad hyper-parameter grid search results')
utils.summarize_gridsearch(cv0)

# Perform a narrow hyperparameter grid search, based on the previous results
print('Perform narrow grid search')
ngram_ranges = [(1,2)]
stop_words = [None]
param_grid = [
    {
        'features__ngram_range': ngram_ranges,
        'features__stop_words': stop_words,
        'classifier__max_depth': [75, 150],
        'classifier__n_estimators': utils.gridspace(20, 200, n=10, log=False)
    }
]
cv1 = sklearn.model_selection.GridSearchCV(pipe, param_grid, 
                                          n_jobs=n_search_jobs, cv=5,
                                          refit=True, verbose=verbosity,
                                          scoring=mcc_scorer)
cv1.fit(train.question, train.coarse_category)
results1 = pd.DataFrame(cv1.cv_results_)
results1.to_csv('results/coarse-grid-search-results-narrow.csv')
print('Narrow hyper-parameter grid search results')
utils.summarize_gridsearch(cv1)

# Perform a narrower hyperparameter grid search, based on the previous
# results
print('Perform narrower grid search')
ngram_ranges = [(1,2)]
stop_words = [None]
param_grid = [
    {
        'features__ngram_range': ngram_ranges,
        'features__stop_words': stop_words,
        'classifier__max_depth': [150],
        'classifier__n_estimators': utils.gridspace(150, 500, n=10, log=True)
    }
]
cv2 = sklearn.model_selection.GridSearchCV(pipe, param_grid, 
                                          n_jobs=n_search_jobs, cv=5,
                                          refit=True, verbose=verbosity,
                                          scoring=mcc_scorer)
cv2.fit(train.question, train.coarse_category)
results2 = pd.DataFrame(cv2.cv_results_)
results2.to_csv('results/coarse-grid-search-results-narrower.csv')
print('Narrower hyper-parameter grid search results')
utils.summarize_gridsearch(cv2)

# Echo results
grid = cv2
print('Best classifier hyperparameters:')
utils.write_fit_params(grid.best_params_, 'results/coarse-model-params.json')
# Save trained model
coarse_classifier = grid.best_estimator_
pickle.dump(coarse_classifier, open('results/coarse-classifier.p', 'wb'))

