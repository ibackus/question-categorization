#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:10:14 2017

@author: ibackus
"""
# General imports
from multiprocessing import cpu_count
import pandas as pd
import json
import numpy as np
import itertools

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
import sklearn.metrics

import utils

# RUN FLAGS
do_train_coarse = True
do_train_fine = True

# Load data
train_filename = 'data/train_5500.label.txt'
test_filename = 'data/test_TREC_10.label.txt'
train = utils.load(train_filename)
test = utils.load(test_filename)
# Set-up numeric labels
labelencoder = preprocessing.LabelEncoder()
train['coarse_label'] = labelencoder.fit_transform(train['coarse_category'])
test['coarse_label'] = labelencoder.transform(test['coarse_category'])
train['fine_label'] = labelencoder.fit_transform(train['fine_category'])
test['fine_label'] = labelencoder.transform(test['fine_category'])

# Get sub-sets of all training and test data according to the coarse category
categories = np.arange(train.coarse_label.max()+1)
subsets = []
for data in (train, test):
    subset = [data[data['coarse_label'] == category] for category in categories]
    subsets.append(subset)
train_subs, test_subs = subsets

# Generate a list of all combinations of categories, up to a max length
category_subsets = []
max_classes = 5
for L in range(1, max_classes + 1):
  for subset in itertools.combinations(categories, L):
    category_subsets.append(subset)
# Now make a look-up table for the index corresponding to a tuple of categories
subset_index = {}
for i, category_subset in enumerate(category_subsets):
    subset_index[category_subset] = i

if do_train_coarse:
    # Coarse classifier
    coarse_classifier = Pipeline([
            ('features', CountVectorizer(ngram_range=(1,2))),
            ('classifier', ExtraTreesClassifier(max_depth=150, random_state=88,
                               n_estimators=200, n_jobs=cpu_count()-1)),
    ])
    # Fit coarse classifier
    print 'Fitting coarse classifier'
    coarse_classifier.fit(train.question, train.coarse_label)


if do_train_fine:
# Fine classifiers
    fine_classifiers = []
    for _ in range(len(category_subsets)):
        fine_classifier = Pipeline([
                ('features', CountVectorizer(ngram_range=(1,2))),
                ('classifier', ExtraTreesClassifier(max_depth=150, random_state=88*2,
                                   n_estimators=400, n_jobs=cpu_count()-1)),
        ])
        fine_classifiers.append(fine_classifier)
    
    # Fit fine classifiers
    print 'Fitting fine classifiers'
    for i, clf in enumerate(fine_classifiers):
        categories = category_subsets[i]
        print "Fitting categories:", categories
        data = pd.concat([train_subs[category] for category in categories])
        clf.fit(data.question, data.fine_label)

def predict_coarse_classes(X):
    """
    Predicts the candidate coarse classes that X could belong to
    """
    # Get classifier probability of belonging to each class
    prob = coarse_classifier.predict_proba(X)
    # Now find the classes required to give cumulative probability greater
    # than 0.95, up to some max number of classes
    ind = (-prob).argsort(axis=1)
    prob_sort = -np.sort(-prob, axis=1)
    cum_prob = np.cumsum(prob_sort, axis=1)
    n_class = np.argmax((cum_prob > 0.95), axis=1) + 1
    n_class[n_class > max_classes] = max_classes
    # This is a bit of a tricky one-liner.  We loop over data points and 
    # find the number of classes we want, then select that many from the
    # sorted class indices
    coarse_classes = [tuple(np.sort(i[0:n])) for n, i in zip(n_class, ind)]
    return coarse_classes

def predict(X):
    """
    
    """
    coarse_classes = predict_coarse_classes(X)
    # Get subset indices
    clf_ind = np.array([subset_index[c] for c in coarse_classes])
    y_pred = []
    for iPoint, iClassifier in enumerate(clf_ind):
        print "Predicting point {} of {}".format(iPoint + 1, len(clf_ind))
        clf = fine_classifiers[iClassifier]
        y_predi = clf.predict(X[[i]])
        y_pred.append(y_predi)
    return np.concatenate(y_pred).flatten()
        