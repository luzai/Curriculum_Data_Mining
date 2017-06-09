from __future__ import division

import time

import keras.backend as K
import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2

import data
import ops,os

from  utils import root_path
os.chdir(root_path)

np.random.seed(13575)

BATCH_SIZE = 10000
nb_users = 6040
nb_items = 3952
DIM = 30
EPOCH_MAX = 10000
DEVICE = "/gpu:0"
tf_graph = tf.get_default_graph()
_sess_config = tf.ConfigProto(allow_soft_placement=True)
_sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=_sess_config, graph=tf_graph)
K.set_session(sess)


def clip(x):
    return np.clip(x, 1.0, 5.0)


def my_mse(pred, gt):
    assert pred.shape == gt.shape
    return np.sqrt(((pred - gt) ** 2).mean())


def my_acc(pred, gt):
    assert pred.shape == gt.shape
    p = pred.ravel().copy()
    g = gt.ravel().copy()
    p = np.round(p)
    return (g == p).sum() / float(len(g))


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def get_data():
    # df=dataio.read_process('ml-20m/ratings.csv',sep=',')
    df = data.read_process('data/train_sub_txt.txt', sep=' ')
    # df = dataio.read_process("ml-1m/ratings.dat", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9) // 2 * 2
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    global nb_users, nb_items, BATCH_SIZE
    USER_NUM = np.array(df.user).max() + 1
    ITEM_NUM = np.array(df.item).max() + 1
    # BATCH_SIZE=min(len(df_train)//2,10000)
    BATCH_SIZE = len(df_train) // 2
    print USER_NUM, ITEM_NUM
    return df_train, df_test


def svd(train, test):
    samples_per_batch = len(train) // BATCH_SIZE

    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["item"],
                                         train["rate"]],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.OneEpochIterator([test["user"],
                                         test["item"],
                                         test["rate"]],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=nb_users, item_num=nb_items, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="tmp_tf/svd/log", graph=sess.graph)
        print("{} {} {}".format("epoch", "train_error", "val_error"))

        pred_train, rate_train = np.array([]), np.array([])

        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates})
            pred_batch = clip(pred_batch)
            pred_train = np.append(pred_train, pred_batch)
            rate_train = np.append(rate_train, rates)
            if (i + 1) % samples_per_batch == 0:
                train_err = my_mse(pred_train, rate_train)
                train_acc = my_acc(pred_train, rate_train)
                pred_train, rate_train = np.array([]), np.array([])

                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items})
                    pred_batch = clip(pred_batch)
                end = time.time()
                test_err = my_mse(pred_batch, rates)
                test_acc = my_acc(pred_batch, rates)

                print("{:3d} train_rmse {:f} test_rmse {:f} train_acc {:f} test_acc {:f}".format(i // samples_per_batch,
                                                                 train_err, test_err,
                                                                 train_acc, test_acc))

                def my_summary(writer, name, data, ind):
                    writer.add_summary(make_scalar_summary(name, data), i)

                my_summary(summary_writer, "train_mse", train_err, i)
                my_summary(summary_writer, "test_mse", test_err, i)
                my_summary(summary_writer, "train_acc", train_acc, i)
                my_summary(summary_writer, "test_acc", test_acc, i)
                start = end


if __name__ == '__main__':
    df_train, df_test = get_data()
    svd(df_train, df_test)
    print("Done!")
