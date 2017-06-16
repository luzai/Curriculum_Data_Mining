import os

import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2

from logger import logger
from  utils import root_path, my_mse, my_acc

os.chdir(root_path)


def clip(x):
    return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def my_summary(writer, name, data, ind):
    writer.add_summary(make_scalar_summary(name, data), ind)


def fc_layer(input, name="fc", activation='relu'):
    with tf.name_scope(name):
        size_in = size_out = input.get_shape().as_list()[-1]
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        if activation.lower == 'relu':
            act = tf.nn.relu(tf.matmul(input, w) + b)
        elif activation.lower() == 'tanh':
            act = tf.nn.tanh(tf.matmul(input, w) + b)
        else:
            act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def corr_layer(in1, in2, name='corr'):
    with tf.name_scope(name):
        size = in1.get_shape().as_list()[-1]
        assert in1.get_shape().as_list() == in2.get_shape().as_list()
        w = tf.Variable(tf.truncated_normal([size, size - 1], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size - 1]), name="B")
        act = tf.nn.relu(tf.matmul(in1, w) + b)

        w2 = tf.Variable(tf.truncated_normal([size, size - 1], stddev=0.1), name="W")
        b2 = tf.Variable(tf.constant(0.1, shape=[size - 1]), name="B")
        act2 = tf.nn.relu(tf.matmul(in2, w2) + b2)

        res = tf.reduce_sum(tf.multiply(act, act2))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("res", res)
        return res


def embedding_layer(user_batch, item_batch, user_num, item_num, dim):
    # with tf.variable_scope('embedding',reuse=True) as scope:
    with tf.name_scope('embedding'):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
        infers = [tf.reduce_sum(tf.multiply(embd_user, embd_item), 1), bias_global, bias_user, bias_item]
        return embd_user, embd_item, infers


from keras.models import Model
from keras.layers import Input, Dense, dot, add
from keras import regularizers


class DeepCF:
    def __init__(self, config):
        self.config = config
        self.build()


    def build(self):
        size_user = self.config.nb_items
        size_item = self.config.nb_users
        user = Input(shape=(size_user,))
        item = Input(shape=(size_item,))
        user_in, item_in = user, item
        res = []
        for layer_ind in range(self.config.layers):
            size_user = max(size_user / 2, 4)
            size_item = max(size_item / 2, 4)
            user = Dense(size_user, activation='tanh',
                         kernel_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1(0.01))(user)
            item = Dense(size_item, activation='tanh',
                         kernel_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1(0.01))(item)
            if layer_ind != self.config.layers - 1:
                res.append(dot([
                    Dense(3)(user),
                    Dense(3)(item)
                ], axes=0))
            else:
                res.append(dot([user, item], axes=0))
        out = add(res)
        model = Model(inputs=[user_in, item_in], outputs=out)
        model.compile(optimizer='adam', loss='rmse')
        self.model=model


class SVD:
    def __init__(self, config):
        self.config = config
        self.build(config.nb_users, config.nb_items, dim=config.dim, reg=config.reg, layers=config.layers)

    def build(self, nb_users, nb_items, dim=5, learning_rate=1e-3, reg=0.1, layers=0):
        with self.config.tf_graph.as_default():
            self.user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
            self.item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
            self.keep_prob = tf.placeholder(tf.float32)

            embd_user, embd_item, infers = embedding_layer(self.user_batch, self.item_batch, nb_users, nb_items, dim)
            embd_user, embd_item = tf.nn.dropout(embd_user, self.keep_prob), tf.nn.dropout(embd_item, self.keep_prob)

            for ind in range(layers):
                embd_user, embd_item = fc_layer(embd_user, name='fc_user_' + str(ind)), fc_layer(embd_item,
                                                                                                 name='fc_item_' + str(
                                                                                                     ind))
                embd_user, embd_item = tf.nn.dropout(embd_user, self.keep_prob), tf.nn.dropout(embd_item,
                                                                                               self.keep_prob)
                infers.append(corr_layer(embd_user, embd_item, name='corr_' + str(ind)))

            for _infer in infers:
                if 'infer' not in locals().keys():
                    infer = _infer
                else:
                    infer = tf.add(infer, _infer)
            self.infer = infer

            self.regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")

            self.rate_batch = tf.placeholder(tf.float32, shape=[None])
            cost_l2 = tf.nn.l2_loss(tf.subtract(self.infer, self.rate_batch))
            penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
            self.cost = tf.add(cost_l2, tf.multiply(self.regularizer, penalty))
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost,
                                                                           global_step=tf.train.get_global_step())

    def optimize(self):
        pass

    def train(self):
        max_test_acc = 0
        iter_train, iter_test = self.config.iter_train, self.config.iter_test
        init_op = tf.global_variables_initializer()
        sess = self.config.sess
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="tmp_tf/svd/log", graph=sess.graph)
        logger.info("{} {} {}".format("epoch", "train_error", "val_error"))

        pred_train, rate_train = np.array([]), np.array([])

        for i in range(self.config.epochs * self.config.samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([self.train_op, self.infer], feed_dict={self.user_batch: users,
                                                                             self.item_batch: items,
                                                                             self.rate_batch: rates,
                                                                             self.keep_prob: self.config.keep_prob})
            pred_batch = clip(pred_batch)
            pred_train = np.append(pred_train, pred_batch)
            rate_train = np.append(rate_train, rates)
            if (i + 1) % self.config.samples_per_batch == 0:
                train_err = my_mse(pred_train, rate_train)
                train_acc = my_acc(pred_train, rate_train)
                pred_train, rate_train = np.array([]), np.array([])

                for users, items, rates in iter_test:
                    pred_batch = sess.run(self.infer, feed_dict={self.user_batch: users,
                                                                 self.item_batch: items,
                                                                 self.keep_prob: 1.})
                    pred_batch = clip(pred_batch)

                test_err = my_mse(pred_batch, rates)
                test_acc = my_acc(pred_batch, rates)
                if test_acc.max() > max_test_acc:
                    max_test_acc = test_acc.max()
                logger.info(
                    "{:3d} train_rmse {:f} test_rmse {:f} train_acc {:f} test_acc {:f}".format(
                        i // self.config.samples_per_batch,
                        train_err, test_err,
                        train_acc, test_acc))

                my_summary(summary_writer, "train_mse", train_err, i)
                my_summary(summary_writer, "test_mse", test_err, i)
                my_summary(summary_writer, "train_acc", train_acc, i)
                my_summary(summary_writer, "test_acc", test_acc, i)
        return max_test_acc

    def predict(self, test_data):
        sess = self.config.sess
        pred_batch = sess.run(self.infer, feed_dict={self.user_batch: test_data[:, 0],
                                                     self.item_batch: test_data[:, 1],
                                                     self.keep_prob: 1.})
        pred_batch = clip(pred_batch)
        return pred_batch
