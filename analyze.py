#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This analyzes the trained extra trees models on the question
classification data set.

The models were trained by coarse_training.py and fine_training.py.

Created on Wed Dec  6 15:32:06 2017

@author: ibackus
"""
# Generic imports
from multiprocessing import cpu_count
import pickle
import pandas as pd
import numpy as np

# Machine learning imports
import sklearn.metrics

# Project specific imports
import utils

# --------------------------------------------------
# Function definitions
# --------------------------------------------------

def load_classifier(filename):
    """
    Loads one of the pickled classifiers
    
    Assumed to be a 2-step sklearn pipeline
    """
    classifier = pickle.load(open(filename,'rb'))
    # Assume 2-step pipeline for these classifiers and update number of 
    # jobs on second step
    # Some classifiers use n_jobs
    classifier.steps[1][1].n_jobs = cpu_count() - 1
    # Some classifiers use nthread
    classifier.steps[1][1].nthread = cpu_count() - 1
    return classifier

def print_results(classifiers, train, test, 
                  labels=['coarse_category', 'fine_category'], refit=False):
    """
    
    """
    accuracy, baseline, mcc_score = [], [], []
    for classifier, label in zip(classifiers, labels):
        y_train = train[label]
        y_test = test[label]
        if refit:
            print("Refitting on:", label)
            classifier.fit(train.question, y_train)
        # Summarize results
        print('\n\nResults summary for:', label)
        y_pred = classifier.predict(test.question)
        print(sklearn.metrics.classification_report(y_test, y_pred))
        # Accuracy
        acc = (y_pred == y_test).mean()
        print('Accuracy:', acc)
        accuracy.append(acc)
        # Baseline accuracy
        descr = y_train.describe()
        base = descr['freq']/float(descr['count'])
        print('Baseline accuracy:', base)
        baseline.append(base)
        # Matthews correlation coefficient score
        mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
        print ('Matthews Correlation Coefficient:', mcc)
        mcc_score.append(mcc)
        
    # Just a final, simple summary
    print('----------------------------------------------------------')
    print('Accuracy summary (name, accuracy, baseline, MCC):')
    for label, acc, base, mcc in zip(labels, accuracy, baseline, mcc_score):
        print(label, acc, base, mcc)


# --------------------------------------------------
# Run script
# --------------------------------------------------

if __name__ == '__main__':
    
    # ------------------------------------
    # Load test/training data
    # ------------------------------------
    train_filename = 'data/train_5500.label'
    test_filename = 'data/TREC_10.label'
    test, train = [utils.load(f) for f in (test_filename, train_filename)]
    
    # ------------------------------------
    # Load classifiers
    # ------------------------------------
    xtratree_filenames = ('results/coarse-classifier.p', 
                          'results/fine-classifier.p')
    xgb_filenames = ('results/xgb-coarse-classifier.p', 
                     'results/xgb-fine-classifier.p')
    xtratrees = [load_classifier(fname) for fname in xtratree_filenames]
    xgbs = [load_classifier(fname) for fname in xgb_filenames]
    
    print('\n----------------------------------------------------------')
    print('Extra-Trees classifiers + bag of words\n')
    print_results(xtratrees, train, test, refit=True)
    
    print('\n----------------------------------------------------------')
    print('XGBoost classifiers + bag of words\n')
    print_results(xgbs, train, test)
    
    # Analyze fine data
    # Count how many examples there were of each category in the training set
    test['pred'] = xgbs[1].predict(test.question) 
    test['correct'] = (test['pred'] == test['fine_category'])
    fine = pd.DataFrame(train.fine_category.value_counts())
    fine.rename(columns={'fine_category':'train-count'})
    # Score the fine-categories
    fine['correct'] = 0
    fine['count'] = 0
    for name, correct in zip(test.fine_category, test.correct):
        row = fine.loc[name]
        row['count'] += 1
        row['correct'] += correct
    fine['accuracy'] = fine['correct']/fine['count']
    # check out coarse categories
    coarse = pd.DataFrame(index=np.unique(train.coarse_category.values))
    coarse['correct'] = 0
    coarse['count'] = 0
    coarse['n_fine_categories'] = 0
    for category in coarse.index:
        mask = np.array([category + ':' in ind for ind in fine.index])
        row = coarse.loc[category]
        row['correct'] = fine['correct'][mask].sum()
        row['count'] = fine['count'][mask].sum()
        row['n_fine_categories'] = mask.sum()
    coarse['accuracy'] = coarse['correct']/coarse['count']
    print("\nFine category results, arranged by coarse category:")
    print(coarse)

