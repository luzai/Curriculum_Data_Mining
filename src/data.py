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
from keras.utils.np_utils import to_categorical


def summary(x, name='defaule'):
    logger.info('{} sumary'.format(name))
    logger.info('min {} max {} mean {} median {}'.format(np.min(x), np.max(x), np.mean(x), np.median(x)))
    logger.info('shape {}'.format(x.shape))


root_path = root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)

os.chdir(root_path)


class Data:
    def __init__(self, name, path=None, sep=None, clean=True, shuffle=True):
        self.shuffle = shuffle
        self.name = name
        self.clean = clean
        self.path = path
        self.sep = sep
        if name == 'train_sub_txt':
            self.path, self.sep = 'data/train_sub_txt.txt', ' '
        self.df_raw = self.get_all_data()
        if clean:
            self.clean_data()
        if self.shuffle:
            self.rand_transform_data()
        self.df_train, self.df_test = self.split_data()

        if name == 'train_sub_txt':
            assert self.nb_users == 285  # and self.nb_items == 1682

    def summary(self):
        logger.info(self.name)
        logger.info('nb_user {} nb_items {} train {} test {}'.format(self.nb_users, self.nb_items, len(self.df_train),
                                                                     len(self.df_test)))
        array_data = self.get_array()
        cnt_users = (array_data != 0).sum(axis=1)
        cnt_items = (array_data != 0).sum(axis=0)
        sparse_ratio = (array_data != 0).sum() / float((np.ones_like(array_data).sum()))
        summary(cnt_users, self.name)
        summary(cnt_items, self.name)
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
        raw_data = np.array(df)[..., :3].astype('float')
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
        summary(cnt_users, self.name)
        summary(cnt_items, self.name)

        keep_items = np.where(np.logical_and(cnt_items > 20.9, cnt_items <= 209))[0]
        self.keep_items_after2before_tuple = (keep_items, np.arange(keep_items.shape[0]))
        self.ori_nb_items = self.nb_items
        array = array[:, keep_items]

        cnt_users = (array != 0).sum(axis=1)
        cnt_items = (array != 0).sum(axis=0)
        summary(cnt_users, self.name)
        summary(cnt_items, self.name)
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

    @staticmethod
    def to_balance_df(df_t):
        df_t = df_t.sort_values(by=['rate']).reset_index(drop=True)
        nums = np.histogram(df_t.rate, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])[0].astype('int32')
        cumnums = np.cumsum(nums)
        start = 0
        for interval, end in zip(nums, cumnums):
            idx = np.random.choice(interval, max(nums) - interval)
            df_t = df_t.append(df_t[start:end].iloc[idx], ignore_index=True)
            start = end
        assert len(df_t) == max(nums) * 5
        df_t = df_t.iloc[np.random.permutation(len(df_t))].reset_index(drop=True)
        verify = np.histogram(df_t.rate, bins=[.5, 1.5, 2.5, 3.5, 4.5, 5.5])[0].astype('int32')
        verify = np.diff(verify)
        assert verify.any() == 0
        return df_t

    def split_data(self, ratio=.9):
        df = self.df_raw
        rows = len(df)
        df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
        split_index = int(rows * ratio) // 2 * 2
        df_train = df[0:split_index]
        df_test = df[split_index:].reset_index(drop=True)
        # not aug here
        # df_train = self.to_balance_df(df_train)

        df_train = df_train.iloc[np.random.permutation(len(df_train))].reset_index(drop=True)
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
        fig.savefig(self.name + '.pdf')
        fig.savefig(self.name + '.png')

    def get_all_test(self, rows=None, cols=None):
        if rows is None or cols is None:
            rows = self.nb_users
            cols = self.nb_items
        x = np.arange(rows)
        y = np.arange(cols)
        yy, xx = np.meshgrid(y, x)
        data = self.get_array()
        return np.array([xx.ravel(), yy.ravel(), data.ravel()]).astype('int32').transpose()

    def make_all_test_batch(self):
        array = self.df_to_array(self.df_raw)
        user_in, item_in = [], []
        rate_out = []
        for user, item, _rate in self.get_all_test():
            # for user, item in zip(range(self.nb_users), range(self.nb_items)):
            user_in_t = array[user, :].copy().ravel()
            user_in_t[item] = 0
            user_in.append(user_in_t)

            item_in_t = array[:, item].copy().ravel()
            item_in_t[user] = 0
            item_in.append(item_in_t)
            assert _rate == array[user, item]
            rate_out.append(array[user, item])

        all_data = (np.array(user_in), np.array(item_in), np.array(rate_out))
        return all_data

    def make_batch(self):
        # assert shape same
        df_train, df_test = self.split_data()
        train_array = self.df_to_array(df_train)
        user_in, item_in = [], []
        rate_out = []
        for user, item in zip(df_train.user, df_train.item):
            user_in_t = train_array[user, :].copy().ravel()
            user_in_t[item] = 0
            user_in.append(user_in_t)

            item_in_t = train_array[:, item].copy().ravel()
            item_in_t[user] = 0
            item_in.append(item_in_t)

            rate_out.append(train_array[user, item])

        # train_data = zip(user_in, item_in, rate_out)
        # train_data.sort(key=lambda x: x[2])
        # df_t = pd.DataFrame(data=train_data, columns=['user', 'item', 'rate'])
        # df_t = self.to_balance_df(df_t)
        # train_data = (np.array(df_t.user.tolist()),
        #               np.array(df_t.item.tolist()),
        #               to_categorical(np.array(df_t.rate) - 1, 5))

        train_data = (np.array(user_in), np.array(item_in), to_categorical(np.array(rate_out) - 1, 5))

        user_in, item_in = [], []
        rate_out = []
        test_array = self.df_to_array(df_test)
        for user, item in zip(df_test.user, df_test.item):
            user_in_t = train_array[user, :].copy().ravel()
            user_in_t[item] = 0
            user_in.append(user_in_t)

            item_in_t = train_array[:, item].copy().ravel()
            item_in_t[user] = 0
            item_in.append(item_in_t)

            rate_out.append(test_array[user, item])

        test_data = (np.array(user_in), np.array(item_in), to_categorical(np.array(rate_out) - 1, 5))

        return train_data, test_data

    def get_origin_array(self, rate, use_gt=False):
        data = self.get_all_test().astype('float')
        if use_gt:
            old_rate = data[:, 2].copy()
            final_rate = np.round(rate.copy()).astype('int32')

            logger.info('original rate and preidct rate distance is {}'.format(
                utils.my_rmse(old_rate[old_rate != 0], rate[old_rate != 0])))
            assert final_rate.shape == old_rate.shape
            final_rate[old_rate != 0] = old_rate[old_rate != 0]
            logger.info('restore the predict rate, distance is {}'.format(
                utils.my_rmse(old_rate[old_rate != 0], final_rate[old_rate != 0])))
        else:
            final_rate = rate
        data[:, 2] = final_rate

        df = pd.DataFrame(data=data, columns=['user', 'item', 'rate'])
        array = self.df_to_array(df)
        assert array.max() <= 5. and array.min() >= 0.
        if self.shuffle:
            array = array[self.users_after2before, :][:, self.items_after2before]
        if self.clean:
            keep_items, ind = self.keep_items_after2before_tuple
            array_true = np.ones((self.nb_users, self.ori_nb_items)) * -1
            array_true[:, :][:, keep_items] = array
            array = array_true
        return array

    def save_res(self, array, name, type='array'):
        if type == 'table':
            df_final = self.array_to_df(array)
            data = np.array(df_final)
            data[:, 0] += 1
            data[:, 1] += 1
            np.savetxt('output/' + name + '.txt', data, fmt='%d %d %.3f')
        else:
            np.savetxt('output/' + name + '.txt', array, fmt='%.18e')


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


import glob

if __name__ == '__main__':

    svd_arrs = []
    deep_cf_arrs = []
    for fn in glob.glob('output/svd_*.txt'):
        array = np.loadtxt(fn)
        svd_arrs.append(array)
    for fn in glob.glob('output/deep_cf_*.txt'):
        array = np.loadtxt(fn)
        deep_cf_arrs.append(array)

    for ind in range(len(svd_arrs)):
        if (svd_arrs[ind] != -1).all():
            complete_arr = svd_arrs[ind]
            del svd_arrs[ind]
            break
    gama = 0.7
    cleaned_arr = np.array(svd_arrs).mean(axis=0) * gama + (1 - gama) * np.array(deep_cf_arrs).mean(axis=0)
    final_arr = cleaned_arr.copy()
    final_arr[final_arr == -1] = complete_arr[final_arr == -1]

    ori_arr = Data('train_sub_txt', clean=False, shuffle=False).get_array()
    final_arr[ori_arr != 0] = ori_arr[ori_arr != 0]

    final_arr2 = final_arr.copy()
    np.savetxt('arr.txt', final_arr2, fmt='%.3f')
    Data('train_sub_txt', clean=False, shuffle=False).save_res(final_arr2, '../res', type='table')
