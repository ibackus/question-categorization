#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a simple wrapper script to train all the models here.

NOTE: This may be SLOW!  This training involves scanning over hyperparamter
space to select good model parameters and can take a long time.  If you
don't want to wait, the trained models are already available and can 
be analyzed by analyze.py

Created on Sat Dec  9 16:14:15 2017

@author: ibackus
"""

print('\nTraining bag of words + extra trees on coarse categories')
exec(open('coarse_training.py').read())

print('\nTraining bag of words + extra trees on fine categories')
exec(open('fine_training.py').read())

print('\nTraining bag of words + XGBoost on coarse+fine categories')
exec(open('xgb_training.py').read())

