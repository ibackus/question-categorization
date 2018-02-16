#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script trains an optimized models on the fine category of the
question classification problem.

The model here is a simple bag of words to extract features with an 
extra-trees ensemble classifier.

A grid search on hyperparameters is performed to determine the best 
hyperparameters, using a 5-fold cross validation and the matthews correlation
coefficient to score models parameters.

The grid parameters were chosen based on information from a series of grid
searches for the same model trained on the coarse categories.  
See coarse_training.py

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
# Train on fine data
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
ngram_ranges = [(1,2)]
stop_words = [None]
param_grid = [
    {
            'features__ngram_range': ngram_ranges,
            'features__stop_words': stop_words,
            'classifier__max_depth': utils.gridspace(2, 200, n=6, log=True),
            'classifier__n_estimators': utils.gridspace(5, 2000, n=20, log=True)
    }
]
cv = sklearn.model_selection.GridSearchCV(pipe, param_grid, 
                                          n_jobs=n_search_jobs, cv=5,
                                          refit=True, verbose=verbosity,
                                          scoring=mcc_scorer)
cv.fit(train.question, train.fine_category)
results0 = pd.DataFrame(cv.cv_results_)
results0.to_csv('results/fine-grid-search-results.csv')
print('Broad hyper-parameter grid search results')
utils.summarize_gridsearch(cv)

# And, finally, let's see the best params!
print('Best classifier hyperparameters:')
utils.write_fit_params(cv.best_params_, 'results/fine-model-params.json')
# Save trained model
fine_classifier = cv.best_estimator_
# The file-size of this fine classifier can get a bit out of control, so to
# save filespace I initiailize a new classifier and copy the classifier 
# parameters over
pipe = Pipeline([
    ('features', CountVectorizer()),
    ('classifier', ExtraTreesClassifier(max_depth=75, random_state=88,
                       n_estimators=50, n_jobs=n_clf_jobs)),
])
# Copy parameters
for i in range(2):
    pipe.steps[i][1].set_params(**fine_classifier.steps[i][1].get_params())
pickle.dump(pipe, open('results/fine-classifier.p', 'wb'))

