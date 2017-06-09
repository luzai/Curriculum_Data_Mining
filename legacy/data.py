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
import utils


def summary(x):
    print np.min(x), np.max(x), np.mean(x)
    print len(np.unique(x)), len(x)


root_path = root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)


class Data(object):
    fn = 'train_sub_txt.txt'
    path = osp.join(root_path, 'data', fn)

    nb_users = 285
    nb_items = 1682

    def __init__(self):
        self.raw_data = np.loadtxt(self.path).astype('int')
        self.sp = self.to_sp()

    def split(self, ratio=.9):
        bak = self.raw_data.copy()
        # np.random.shuffle(self.raw_data)
        raw_data = self.raw_data.copy()
        self.raw_data = bak

        trains = int(raw_data.shape[0] * ratio)
        self.trainset = raw_data[:trains]
        self.testset = raw_data[trains:]

    def to_sp(self):
        # todo note convert back to 1 based
        users = self.raw_data[:, 0] - 1
        items = self.raw_data[:, 1] - 1
        scores = self.raw_data[:, 2]

        from scipy.sparse import coo_matrix

        sparse_mat = coo_matrix((scores, (users, items)), shape=(self.nb_users, self.nb_items))
        # mat = sparse_mat.toarray()
        return sparse_mat

    def vis_data(self):
        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111)
        ax.matshow(self.sp.toarray())
        plt.colorbar()
        plt.axis('off')
        fig.show()
        fig.savefig('t.pdf')


def get_all_data(rows, cols):
    x = np.arange(rows)
    y = np.arange(cols)
    yy, xx = np.meshgrid(y, x)
    data = np.zeros(xx.shape)
    return np.array([xx.ravel(), yy.ravel(), data.ravel()])
