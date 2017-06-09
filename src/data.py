from __future__ import division

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path as osp

import utils

from logger import logger


def summary(x):
    print np.min(x), np.max(x), np.mean(x)
    print len(np.unique(x)), len(x)


root_path = root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)
import os

os.chdir(root_path)


class Data:
    def __init__(self, name):
        self.name = name
        if name == 'train_sub_txt':
            self.path, self.sep = 'data/train_sub_txt.txt', ' '
        elif name == 'ml-20m':
            self.path, self.sep = 'data/ml-20m/ratings.csv', ','
        elif name == 'ml-1m':
            self.path, self.sep = "data/ml-1m/ratings.dat", "::"
        self.df_train, self.df_test = self.get_split_data()

        if name == 'train_sub_txt':
            assert self.nb_users == 285 and self.nb_items == 1682

    def summary(self):
        logger.info('nb_user {} nb_items {} train {} test {}'.format(self.nb_users, self.nb_items, len(self.df_train),
                                                                     len(self.df_test)))
        array_data = self.get_sp().toarray()
        cnt_users = (array_data != 0).sum(axis=1)
        cnt_items = (array_data != 0).sum(axis=0)
        sparse_ratio= (array_data!=0).sum()/float((np.ones_like(array_data).sum()))
        logger.info('nb_users {} users '.format(cnt_users.shape))
        logger.info('nb_item {} items '.format(cnt_items.shape))
        logger.info('sparse_ratio {}'.format(sparse_ratio))

    @staticmethod
    def read_process(filname, sep="\t"):
        col_names = ["user", "item", "rate", "st"]
        df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
        df["user"] -= 1
        df["item"] -= 1
        for col in ("user", "item"):
            df[col] = df[col].astype(np.int32)
        df["rate"] = df["rate"].astype(np.float32)
        logger.debug(df.head())
        return df

    def clean_data(self):
        pass

    def get_all_data(self):
        df = self.read_process(self.path, sep=self.sep)
        rows = len(df)
        df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
        return df

    def get_split_data(self):
        df = self.get_all_data()
        rows = len(df)
        split_index = int(rows * 0.9) // 2 * 2
        df_train = df[0:split_index]
        df_test = df[split_index:].reset_index(drop=True)
        nb_users = np.array(df.user).max() + 1
        nb_items = np.array(df.item).max() + 1
        self.batch_size = min(len(df_train) // 2, 900188 // 4)
        self.nb_users, self.nb_items = nb_users, nb_items
        return df_train, df_test

    def split(self, ratio=.9):
        bak = self.raw_data.copy()
        # np.random.shuffle(self.raw_data)
        raw_data = self.raw_data.copy()
        self.raw_data = bak

        trains = int(raw_data.shape[0] * ratio)
        self.trainset = raw_data[:trains]
        self.testset = raw_data[trains:]

    def get_sp(self):
        df = self.get_all_data()
        raw_data = np.array(df)[..., :3].astype('int32')
        users = raw_data[:, 0]
        items = raw_data[:, 1]
        scores = raw_data[:, 2]
        from scipy.sparse import coo_matrix
        sparse_mat = coo_matrix((scores, (users, items)), shape=(self.nb_users, self.nb_items))
        return sparse_mat

    def vis_data(self):
        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111)
        ax.matshow(self.get_sp().toarray())
        plt.colorbar()
        plt.axis('off')
        fig.show()
        fig.savefig('t.pdf')

    def get_all_test(self, rows=None, cols=None):
        if rows is None or cols is None:
            rows = self.nb_users
            cols = self.nb_items
        x = np.arange(rows)
        y = np.arange(cols)
        yy, xx = np.meshgrid(y, x)
        data = self.get_sp().toarray()
        return np.array([xx.ravel(), yy.ravel(), data.ravel()]).astype('int32')

    def save_res(self, rate):
        data = self.get_all_test()
        data[0, :] += 1
        data[1, :] += 1
        old_rate = data[2, :].copy()
        final_rate = np.round(rate.copy()).astype('int32')

        logger.info(utils.my_mse(old_rate[old_rate != 0], rate[old_rate != 0]))
        final_rate[old_rate != 0] = old_rate[old_rate != 0]
        logger.info(utils.my_mse(old_rate[old_rate != 0], final_rate[old_rate != 0]))

        data[2, :] = final_rate

        np.savetxt('output/res.txt', data.transpose(), fmt='%d')


class ShuffleIterator(object):
    """
    Randomly generate batches
    """

    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochIterator(ShuffleIterator):
    """
    Sequentially generate one-epoch batches, typically for test data
    """

    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]


if __name__ == '__main__':
    for name in ['train_sub_txt', 'ml-1m']:  # 'ml-20m'
        data = Data(name)
        data.summary()
