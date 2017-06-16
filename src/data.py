from __future__ import division

import matplotlib

matplotlib.use("TkAgg")
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path as osp
import utils
from logger import logger
from scipy.sparse import coo_matrix


def summary(x):
    print np.min(x), np.max(x), np.mean(x), np.median(x)
    print len(np.unique(x)), len(x)


root_path = root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)

os.chdir(root_path)


class Data:
    shuffle = True

    def __init__(self, name, clean=True):
        self.name = name
        self.clean = clean
        if name == 'train_sub_txt':
            self.path, self.sep = 'data/train_sub_txt.txt', ' '
        elif name == 'ml-20m':
            self.path, self.sep = 'data/ml-20m/ratings.csv', ','
        elif name == 'ml-1m':
            self.path, self.sep = "data/ml-1m/ratings.dat", "::"
        self.df_raw = self.get_all_data()
        if clean:
            self.clean_data()
        if self.shuffle:
            self.rand_transform_data()
        self.df_train, self.df_test = self.split_data()

        if name == 'train_sub_txt':
            assert self.nb_users == 285  # and self.nb_items == 1682

    def summary(self):
        logger.info('nb_user {} nb_items {} train {} test {}'.format(self.nb_users, self.nb_items, len(self.df_train),
                                                                     len(self.df_test)))
        array_data = self.get_array()
        cnt_users = (array_data != 0).sum(axis=1)
        cnt_items = (array_data != 0).sum(axis=0)
        sparse_ratio = (array_data != 0).sum() / float((np.ones_like(array_data).sum()))
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
        df = df[['user', 'item', 'rate']]
        logger.debug(df.head())

        return df

    @staticmethod
    def df_to_array(df):
        raw_data = np.array(df)[..., :3].astype('int32')
        users = raw_data[:, 0]
        items = raw_data[:, 1]
        scores = raw_data[:, 2]
        nb_users = np.array(df.user).max() + 1
        nb_items = np.array(df.item).max() + 1
        sparse_mat = coo_matrix((scores, (users, items)), shape=(nb_users, nb_items))
        return sparse_mat.toarray()

    @staticmethod
    def array_to_df(array):
        sparse_mat = coo_matrix(array)
        df = pd.DataFrame()
        df['user'] = sparse_mat.row
        df['item'] = sparse_mat.col
        df['rate'] = sparse_mat.data
        return df

    def clean_data(self):
        array = self.get_array()
        cnt_users = (array != 0).sum(axis=1)
        cnt_items = (array != 0).sum(axis=0)
        keep_items = np.nonzero(cnt_items > 20.9)[0]
        self.keep_items_after2before_tuple = (keep_items, np.arange(keep_items.shape[0]))
        # self.omit_items = np.nonzero(cnt_items < 20.9)[0]
        array = array[:, keep_items]
        self.df_raw = self.array_to_df(array)
        self.nb_users, self.nb_items = array.shape

    def rand_transform_data(self):
        array = self.get_array()
        users_ind_before2after = np.random.permutation(np.arange(self.nb_users))
        self.users_after2before = np.argsort(users_ind_before2after)

        items_ind_before2after = np.random.permutation(np.arange(self.nb_items))
        self.items_after2before = np.argsort(items_ind_before2after)
        array = array[users_ind_before2after, :][:, items_ind_before2after]
        self.df_raw = self.array_to_df(array)

    def get_all_data(self):
        df = self.read_process(self.path, sep=self.sep)
        rows = len(df)
        df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
        nb_users = np.array(df.user).max() + 1
        nb_items = np.array(df.item).max() + 1
        self.nb_users, self.nb_items = nb_users, nb_items
        return df

    def split_data(self):
        df = self.df_raw
        rows = len(df)
        split_index = int(rows * 0.9) // 2 * 2
        df_train = df[0:split_index]
        df_test = df[split_index:].reset_index(drop=True)
        self.batch_size = min(len(df_train) // 2, 900188 // 4)

        return df_train, df_test

    def get_array(self):
        return self.df_to_array(self.df_raw)

    def vis_data(self):
        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111)
        ax.matshow(self.get_array())
        # plt.colorbar()
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
        data = self.get_array()
        return np.array([xx.ravel(), yy.ravel(), data.ravel()]).astype('int32').transpose()

    def get_origin_array(self, rate):
        data = self.get_all_test()
        old_rate = data[:, 2].copy()
        final_rate = np.round(rate.copy()).astype('int32')

        logger.info('original rate and preidct rate distance is {}'.format(
            utils.my_mse(old_rate[old_rate != 0], rate[old_rate != 0])))
        final_rate[old_rate != 0] = old_rate[old_rate != 0]
        logger.info('restore the predict rate, distance is {}'.format(
            utils.my_mse(old_rate[old_rate != 0], final_rate[old_rate != 0])))

        data[:, 2] = final_rate

        df = pd.DataFrame(data=data, columns=['user', 'item', 'rate'])
        array = self.df_to_array(df)
        if self.shuffle:
            array = array[self.users_after2before, :][:, self.items_after2before]
        if self.clean:
            keep_items, ind = self.keep_items_after2before_tuple
            nb_items_true = keep_items.max() + 1
            array_true = np.ones((self.nb_users, nb_items_true)) * -1
            array_true[:, :][:, keep_items] = array
            array = array_true
        return array

    def save_res(self, array):
        df_final = self.array_to_df(array)
        data = np.array(df_final)
        data[:, 0] += 1
        data[:, 1] += 1

        np.savetxt('output/res.txt', data, fmt='%d')


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
    for name in[ 'train_sub_txt' ,'ml-1m']:
        data = Data(name,clean=True)
        data.summary()
        data.vis_data()
    input()