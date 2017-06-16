import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import tensorflow as tf
import keras.backend as K
import multiprocessing as mp

root_path = root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)
import os

os.chdir(root_path)


def my_mse(pred, gt):
    assert pred.shape == gt.shape
    return np.sqrt(((pred - gt) ** 2).mean())


def my_acc(pred, gt):
    assert pred.shape == gt.shape
    p = pred.ravel().copy()
    g = gt.ravel().copy()
    p = np.round(p)
    return (g == p).sum() / float(len(g))


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


def my_plot(x, y=None, name='default'):
    def _my_plot(x, y=None, name='default'):
        fig, ax = plt.subplots()
        if y is not None:
            plt.plot(x, y, '^')
        else:
            plt.plot(x)
        ax.set_title(name)
        fig.canvas.set_window_title(name)
        plt.show()

    mp.Process(target=_my_plot, args=(x, y, name)).start()


def import_tf():
    tf_graph = tf.get_default_graph()
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=_sess_config, graph=tf_graph)
    K.set_session(sess)
    return tf


import os, subprocess


def to_single_dir():
    os.chdir(root_dir)
    for parent, dirnames, filenames in os.walk('tmp_tf'):
        filenames = sorted(filenames)
        if len(filenames)==1:
            continue
        for ind, fn in enumerate(filenames):
            subprocess.call(('mkdir -p ' + parent + '/' + str(ind)).split())
            subprocess.call(('mv ' + parent + '/' + fn + ' ' + parent + '/' + str(ind) + '/').split())
        print parent, filenames


if __name__ == "__main__":
    to_single_dir()
