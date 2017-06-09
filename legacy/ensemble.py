from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import os.path as osp
import numpy as np
import matplotlib, sys, os, \
    glob, cPickle, scipy, \
    argparse, errno, json, \
    copy, re, time, imp, datetime, \
    cv2, logging

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path as osp
import scipy.io as sio

from pprint import pprint
import subprocess
import multiprocessing as mp
import Utils

from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise import evaluate, print_perf
import random

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import GridSearch


def summary(x):
    print(np.min(x), np.max(x), np.mean(x))
    print(len(np.unique(x)), len(x))


root_path = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)
fn = 'train_sub_txt.txt'
file_path = osp.join(root_path, 'data', fn)

nb_users = 285
nb_items = 1682

reader = Reader(line_format='user item rating', sep=' ')

data = Dataset.load_from_file(file_path, reader=reader)
raw_ratings = data.raw_ratings

# shuffle ratings if you want
random.shuffle(raw_ratings)

# A = 90% of the data, B = 10% of the data
threshold = int(.9 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = A_raw_ratings  # data is now the set A
data.split(n_folds=3)

# Select your best algo with grid search.
print('Grid Search...')
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=0)
grid_search.evaluate(data)

algo = grid_search.best_estimator['RMSE']

# retrain on the whole set A
trainset = data.build_full_trainset()
algo.train(trainset)

# Compute biased accuracy on A
predictions = algo.test(trainset.build_testset())
print('Biased accuracy on A,', end='   ')
accuracy.rmse(predictions)

# Compute unbiased accuracy on B
testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
predictions = algo.test(testset)
print('Unbiased accuracy on B,', end=' ')
accuracy.rmse(predictions)
