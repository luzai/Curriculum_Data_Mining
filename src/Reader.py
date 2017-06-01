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
import  multiprocessing as mp
import Utils


def summary(x):
    print np.min(x), np.max(x), np.mean(x)
    print len(np.unique(x)), len(x)

root_path = root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)
fn = 'train_sub_txt.txt'
path = osp.join(root_path, 'data', fn)

nb_users=285
nb_items=1682

data = np.loadtxt(path).astype('int')

print data
users = data[:, 0] -1
items = data[:, 1] -1
scores = data[:, 2]
for i in range(3):
    summary(data[:, i])
# Utils.my_plot(users,name='users')
# Utils.my_plot(items)

from scipy.sparse import coo_matrix

sparse_mat=coo_matrix((scores, (users, items)),shape=(nb_users,nb_items))
mat=sparse_mat.toarray()

fig=plt.figure(figsize=(30,30))
ax=fig.add_subplot(111)
ax.matshow(mat)
# plt.colorbar()
plt.axis('off')
fig.show()
fig.savefig('t.pdf')
