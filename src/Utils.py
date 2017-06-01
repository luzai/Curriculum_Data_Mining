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

def my_imshow(img, cmap=None, block=False, name='default'):
    if block:
        fig, ax = plt.subplots()
        if len(img.shape) == 3 and img.shape[2] == 3 and img.max() > 2.:
            img = img.astype('uint8')
        ax.imshow(img, cmap)
        ax.set_title(name)
        fig.canvas.set_window_title(name)
        plt.show()
    else:
        import multiprocessing
        multiprocessing.Process(target=my_imshow, args=(img, cmap, True, name)).start()
def my_plot(x,y=None,name='default'):
    def _my_plot(x,y=None,name='default'):
        fig, ax = plt.subplots()
        if y is not  None:
            plt.plot(x,y)
        else:
            plt.plot(x)
        ax.set_title(name)
        fig.canvas.set_window_title(name)
        plt.show()
    mp.Process(target=_my_plot,args=(x,y,name)).start()