import os

import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2

from logger import logger
from  utils import root_path, my_rmse, my_acc, mkdir_p

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


from keras.models import Model, load_model
from keras.layers import Input, Dense, add, multiply, Activation, Dropout, BatchNormalization, regularizers
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint


class DeepCF:
    def __init__(self, config, model_name, size=None):
        self.config = config
        self.model_name = model_name
        if config.layers < 2:
            config.layers = 2
        if size is None:
            config.layers = 4
            size = [300, 200, 200]
        self.build(size)

    def build(self, size):
        assert self.config.layers - 1 == len(size)
        size_user = self.config.nb_items
        size_item = self.config.nb_users
        user = Input(shape=(size_user,))
        item = Input(shape=(size_item,))
        user_in, item_in = user, item
        res = []
        for layer_ind in range(self.config.layers):
            if layer_ind != self.config.layers - 1:
                # size_user = max(size_user // 10, 5)
                # size_item = max(size_item // 10, 5)
                size_user = size_item = size[layer_ind]
                user = Dense(size_user,
                             # kernel_regularizer=regularizers.l2(0.01),
                             # activity_regularizer=regularizers.l1(0.01)
                             )(user)
                user = BatchNormalization()(user)
                user = Activation('tanh')(user)
                user = Dropout(1 - self.config.keep_prob)(user)

                item = Dense(size_item,
                             # kernel_regularizer=regularizers.l2(0.01),
                             # activity_regularizer=regularizers.l1(0.01)
                             )(item)
                item = BatchNormalization()(item)
                item = Activation('tanh')(item)
                item = Dropout(1 - self.config.keep_prob)(item)

                res.append(multiply([
                    Dense(5, activation='relu')(user),
                    Dense(5, activation='relu')(item)
                ]))
            else:
                size_user, size_item = 5, 5
                user = Dense(size_user,
                             # kernel_regularizer=regularizers.l2(0.01),
                             # activity_regularizer=regularizers.l1(0.01),
                             name='f_fc_1'
                             )(user)
                user = BatchNormalization()(user)
                user = Activation('relu')(user)
                user = Dropout(1 - self.config.keep_prob)(user)

                item = Dense(size_item,
                             # kernel_regularizer=regularizers.l2(0.01),
                             # activity_regularizer=regularizers.l1(0.01),
                             name='f_fc_2'
                             )(item)
                item = BatchNormalization()(item)
                item = Activation('relu')(item)
                item = Dropout(1 - self.config.keep_prob)(item)
                res.append(multiply([user, item]))

        out = Activation('softmax')(add(res))

        model = Model(inputs=[user_in, item_in], outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # SGD(1e-6) #rmsprop
        # model.summary()
        self.model = model

    def train(self):
        train_data, test_data = self.config.data.make_batch()

        model_name_t = 'output/' + self.model_name + '.h5'
        hist = self.model.fit(x=list(train_data[0:2]),
                              y=train_data[2],
                              validation_data=(list(test_data[0:2]), test_data[2]),
                              batch_size=self.config.batch_size,
                              epochs=self.config.epochs,
                              callbacks=[
                                  TensorBoard(log_dir='tmp_tf/',
                                              # histogram_freq=10,
                                              # write_images=True,
                                              # write_grads=True,
                                              # embeddings_freq=10,
                                              # embeddings_layer_names=['f_fc_1','f_fc_2']
                                              ),
                                  # EarlyStopping(monitor='val_acc', min_delta=-0.012, patience=16, verbose=1),
                                  ModelCheckpoint(model_name_t, monitor='val_acc', save_best_only=True)
                              ],
                              verbose=2)
        self.model.load_weights(model_name_t)
        return hist.history['val_acc'][-1], max(hist.history['val_acc'])

    def predict(self, data, soft=True):
        pred = self.model.predict(x=list(data[:2]), batch_size=self.config.batch_size)
        # print 'loss and acc is', self.model.evaluate(x=list(data[:2]),y=data[2],batch_size=14178)
        if not soft:
            return np.argmax(pred, axis=1) + 1
        else:
            tmp = np.ones_like(pred)
            for ind in range(tmp.shape[1]):
                tmp[:, ind] *= (ind + 1)
            return np.multiply(tmp, pred).sum(axis=1)


class SVD:
    def __init__(self, config, model_name):
        self.model_name = model_name
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
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cost,
                                                                  global_step=tf.train.get_global_step())
            # GradientDescentOptimizer

    def train(self):
        early_stop = MyEarlyStop(min_delta=-0.012, patience=16)
        checkpoint = MyCheckPoint(name=self.model_name)
        max_test_acc, wait = 0, 0
        iter_train, iter_test = self.config.iter_train, self.config.iter_test
        init_op = tf.global_variables_initializer()
        sess = self.config.sess
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="tmp_tf/svd/log", graph=sess.graph)
        pred_train, rate_train = np.array([]), np.array([])

        for global_step in range(self.config.epochs * self.config.samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([self.train_op, self.infer], feed_dict={self.user_batch: users,
                                                                             self.item_batch: items,
                                                                             self.rate_batch: rates,
                                                                             self.keep_prob: self.config.keep_prob})
            pred_batch = clip(pred_batch)
            pred_train = np.append(pred_train, pred_batch)
            rate_train = np.append(rate_train, rates)
            if (global_step + 1) % self.config.samples_per_batch == 0:
                train_err = my_rmse(pred_train, rate_train)
                train_acc = my_acc(pred_train, rate_train)
                pred_train, rate_train = np.array([]), np.array([])

                test_err, test_acc = self.evaluate(iter_test)

                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    checkpoint.judge_save(test_acc, sess, global_step)
                if wait % 200 == 0:
                    logger.info(
                        "{:3d} train_rmse {:f} test_rmse {:f} train_acc {:f} test_acc {:f}".format(
                            global_step // self.config.samples_per_batch,
                            train_err, test_err,
                            train_acc, test_acc))
                wait += 1
                my_summary(summary_writer, "train_mse", train_err, global_step)
                my_summary(summary_writer, "test_mse", test_err, global_step)
                my_summary(summary_writer, "train_acc", train_acc, global_step)
                my_summary(summary_writer, "test_acc", test_acc, global_step)
                if early_stop.judge_stop(test_acc):
                    logger.critical('svd is early stoped at {} '.format(global_step // self.config.samples_per_batch))
                    break
        test_err, test_acc = self.evaluate(iter_test)
        logger.critical('the last test rmse and acc is {} {}'.format(test_err, test_acc))
        self.config.reset()
        sess = self.config.sess
        new_saver = tf.train.import_meta_graph(checkpoint.get_meta_name())
        new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint.path))
        test_err, test_acc = self.evaluate(iter_test)
        logger.critical('the best model test rmse and acc is {} {}'.format(test_err, test_acc))

        return test_acc, max_test_acc

    def evaluate(self, iter_test):
        sess = self.config.sess
        for users, items, rates in iter_test:
            pred_batch = sess.run(self.infer, feed_dict={self.user_batch: users,
                                                         self.item_batch: items,
                                                         self.keep_prob: 1.})
            pred_batch = clip(pred_batch)

        test_err = my_rmse(pred_batch, rates)
        test_acc = my_acc(pred_batch, rates)
        return test_err, test_acc

    def predict(self, test_data):

        sess = self.config.sess
        pred_batch = sess.run(self.infer, feed_dict={self.user_batch: test_data[:, 0],
                                                     self.item_batch: test_data[:, 1],
                                                     self.keep_prob: 1.})
        pred_batch = clip(pred_batch)
        return pred_batch


import glob


class MyCheckPoint:
    def __init__(self, name):
        self.best = 0
        self.path = 'output/' + name
        self.prefix = self.path + '/svd'

    def get_meta_name(self):
        name = glob.glob(self.prefix + '*.meta')
        assert len(name) == 1
        return name[0]

    def judge_save(self, acc, sess, global_step):
        if global_step <= 3000 and acc <= 0.35:
            tol = 0.1
        elif acc <= 0.4:
            tol = 0.06
        elif acc < 0.44:
            tol = 0.02
        else:
            tol = 0

        if acc > self.best + tol or global_step <= 1:
            logger.info('save model with acc {} better than {}'.format(acc, self.best))
            self.best = acc
            mkdir_p(self.path, delete=True)
            saver = tf.train.Saver()
            saver.save(sess, self.prefix, global_step=global_step)


class MyEarlyStop:
    def __init__(self, min_delta, patience=10):
        # decrapted
        self.min_delta = min_delta
        self.patience = patience
        self.best = 0
        self.wait = 0

    def judge_stop(self, acc):
        # print acc, self.best, self.wait
        return False
        if acc - self.min_delta > self.best:
            self.best = max(acc, self.best)
            self.wait = 0
        else:
            if self.wait >= self.patience:
                return True
            self.wait += 1
        return False


class RandomGuess:
    def predict(self, data, num=4):
        data_len = data.shape[0]
        return np.ones((data_len,)) * num


if __name__ == '__main__':
    from config import Config

    config = Config('train_sub_txt', dim=100, epochs=10000, layers=4, reg=.02, keep_prob=.85, clean=True)

    deep_cf = DeepCF(config, 'deep_cf')
    deep_cf.train()

    print deep_cf.predict(config.data.make_all_test_batch())
