#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:22:28 2017

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
do_train_coarse = False
do_train_fine = False

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
# Get fine to coarse mapping
fine2coarse = dict(zip(train['fine_label'], train['coarse_label']))
n_per_coarse = np.zeros(test.coarse_label.max()+1)
for fine, coarse in fine2coarse.iteritems():
    n_per_coarse[coarse] += 1

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
    # Fine classifier
    fine_classifier = Pipeline([
            ('features', CountVectorizer(ngram_range=(1,2))),
            ('classifier', ExtraTreesClassifier(max_depth=200, random_state=88,
                               n_estimators=777, n_jobs=cpu_count()-1)),
    ])
    # Fit fine classifier
    print 'Fitting fine classifier'
    fine_classifier.fit(train.question, train.fine_label)

fine_prob = fine_classifier.predict_proba(test.question)
coarse_prob = coarse_classifier.predict_proba(test.question)

def joint_prob(fine, coarse, coarse_weight=0.3):
    w = coarse_weight
    prob = 0*fine
    for ifine, icoarse in fine2coarse.iteritems():
        prob[:, ifine] = coarse[:, icoarse]/n_per_coarse[icoarse]
    prob = (1 - w + w*prob)*fine
    return prob
    