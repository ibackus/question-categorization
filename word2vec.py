#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a very simple, proof of concept attempt at using word2vec as a feature
extractor for the question classification problem.

The naive word2vec, with a corpus trained on the questions only, performs
very poorly compared to the bag of words approach, so it is abandoned.
Using a more advanced pre-trained word2vec model might perform significantly
better.

Created on Wed Dec  6 20:16:32 2017

@author: ibackus
"""
# General imports
from multiprocessing import cpu_count
import numpy as np

# ML imports
import gensim
import xgboost

# Project imports
import utils

# Load data
train_filename = 'data/train_5500.label'
test_filename = 'data/TREC_10.label'
stoplist = set('for a of the and to in'.split())
train = utils.load(train_filename, stoplist)
test = utils.load(test_filename, stoplist)

# Generate the word2vec model
w2v = gensim.models.Word2Vec(train.split_question, workers=cpu_count(),
                             min_count=5, size=100)
model = w2v.wv
zeros = np.zeros(w2v.vector_size)

# Get a vector representation of the data
def transform(sentences):
    """
    A simple method to transform sentences to a vector representation.  This
    is done by taking the mean of the word2vec representation of all words
    in the sentences.  Words not in the model corpus are treated as a zero
    vector.
    """
    vec = np.array([np.mean([model[w] if (w in model.vocab) else zeros \
        for w in sentence], axis=0) for sentence in sentences])
    return vec

x_train = transform(train.split_question)
x_test = transform(test.split_question)

# Fit a simple classifier
clf = xgboost.XGBClassifier(max_depth=30, n_estimators=200)
clf.fit(x_train, train.coarse_category)
y_pred = clf.predict(x_test)
score = (y_pred == test.coarse_category).mean()
print('Word2Vec + XGBoost accuracy:', score)

