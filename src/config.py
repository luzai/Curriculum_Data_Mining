import tensorflow as tf

import data


class Config(object):
    def __init__(self, name='train_sub_txt', epochs=500, dim=5, reg=.1, layers=0, keep_prob=.9,clean=True):
        self.name = name
        self.epochs = epochs
        self.dim = dim
        self.reg = reg
        self.layers = layers
        self.keep_prob = keep_prob

        self.tf_graph = tf.get_default_graph()
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config, graph=self.tf_graph)

        self.data = data.Data(name=name,clean=clean)
        self.batch_size = self.data.batch_size
        self.samples_per_batch = len(self.data.df_train) // self.batch_size
        self.nb_users = self.data.nb_users
        self.nb_items = self.data.nb_items

        self.iter_train = data.ShuffleIterator([self.data.df_train["user"],
                                                self.data.df_train["item"],
                                                self.data.df_train["rate"]],
                                               batch_size=self.batch_size)
        self.iter_test = data.OneEpochIterator([self.data.df_test["user"],
                                                self.data.df_test["item"],
                                                self.data.df_test["rate"]],
                                               batch_size=self.batch_size)
