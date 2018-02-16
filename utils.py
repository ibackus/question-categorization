#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains some simple utility functions for the question classification
problem.

Created on Wed Dec  6 13:58:07 2017

@author: ibackus
"""
import pandas as pd
import numpy as np
import json

# Data loader
def load(path, stoplist=[], comments='#'):
    """
    Loads the question data sets
    
    Parameters
    ----------
    path : str
        Path to the data set
    stoplist : list
        A list of words to ignore
    comments : str
        Comment character.  Lines beginning with this character are ignored
        
    Returns
    -------
    data : pandas DataFrame
        Returns the data, with questions formatted (remove punctuation) and
        with questions split into a list of words (useful for e.g. Word2Vec)
    """
    rows = []
    with open(path, "r", encoding='latin-1') as data_in:
        for line in data_in:
            line = line.strip().strip('?').lower()
            if (line[0] == comments):
                continue
            fine_category, question = line.split(None, 1)
            coarse_category, _ = fine_category.split(":")
            question = question.strip()
            rows.append({
                "question": question,
                "fine_category": fine_category,
                "coarse_category": coarse_category
            })
    data = pd.DataFrame(rows)
    # Remove punctuation and numbers
    data['question'].replace('[!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]',' ',inplace=True,regex=True)
    data['split_question'] = [[w for w in q.split() if w not in stoplist] for q in data.question]
    for i, q in enumerate(data['split_question']):
        if len(q) == 0:
            data['split_question'][i] = ['']
    return data

def gridspace(x0, x1, n=50, log=False, integer=True):
    """
    Generates numbers between x0 and x1, evenly spaced in linear or log space,
    floats or integers
    """
    if log:
        x0, x1 = (np.log(x0), np.log(x1))
    y = np.linspace(x0, x1, n)
    if log:
        y = np.exp(y)
    if integer:
        y = np.round(y).astype(int)
    return y

def summarize_gridsearch(cv):
    """
    Print a summary of grid search results.
    
    Prints the 10 best grid search parameters and their scores
    
    Parameters
    ----------
    cv : GridSearchCV instance
        A search instance from sklearn.
    """
    results = pd.DataFrame(cv.cv_results_)
    # Now look at the 10 best results to get a sense of the parameters
    bestind = (-results['mean_test_score']).argsort().values
    best = results.iloc[bestind[0:10]]
    # Let's get a list of all keys used in the param grid search
    param_grid_keys = set(cv.param_grid[0].keys())
    for param in cv.param_grid[1:]:
        param_grid_keys.union(list(param.keys()))
    param_grid_keys = ['param_' + key for key in list(param_grid_keys)]
    summary_keys = ['mean_test_score'] + param_grid_keys
    print('Best parameters summary for broad hyperparameter search')
    print(best[summary_keys])
    
def predict_coarse_classes(prob, max_classes=5, cum_prob=0.95):
    """
    Predicts the candidate coarse classes that points could belong to based
    on a coarse classifier.
    
    This was used to test out a hierarchical classifier.
    """
    # Now find the classes required to give cumulative probability greater
    # than 0.95, up to some max number of classes
    ind = (-prob).argsort(axis=1)
    prob_sort = -np.sort(-prob, axis=1)
    cum_prob = np.cumsum(prob_sort, axis=1)
    n_class = np.argmax((cum_prob > cum_prob), axis=1) + 1
    n_class[n_class > max_classes] = max_classes
    # This is a bit of a tricky one-liner.  We loop over data points and 
    # find the number of classes we want, then select that many from the
    # sorted class indices
    coarse_classes = [(np.sort(i[0:n])) for n, i in zip(n_class, ind)]
    return coarse_classes

def xgb_transformer(Transform, *args, **kwargs):
    """
    The xgboost package requires CSC matrices as input.  This creates a 
    Transform instance, passing *args and **kwargs to the constructor, and
    over-rides the 'transform' method to make it return a CSC matrix.
    
    This is very hacky and is only a temporary quick fix until xgboost fixes
    issue #1238
    
    Transform should be an sklearn style class with fit and transform methods.
    """
    bagger = Transform(*args, **kwargs)
    bagger._transform = bagger.transform
    bagger.transform = lambda *a, **k : bagger._transform(*a, **k).tocsc()
    return bagger

def write_fit_params(params, fname, verbose=True):
    """
    A simple utility to write and echo the best fit params in a readable(ish)
    format.
    
    This will convert numpy types to python types in place!
    
    Basically, JSON can't handle some numpy types, so this takes care of that
    
    If verbose, will also echo to stdout
    """
    for k, v in params.items():
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.float):
            v = float(v)
        params[k] = v
    serial = json.dumps(params, indent=2, sort_keys=True)
    if verbose:
        print(serial)
    with open(fname, 'w') as f:
        f.write(serial)

