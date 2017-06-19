import os, subprocess, multiprocessing
import matplotlib
import numpy as np
import os.path as osp

matplotlib.use("TkAgg")
root_path = root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)
os.chdir(root_path)
subprocess.call('./clean.sh')


def client(queue, method, clean, epochs):
    from utils import root_path, my_acc, my_rmse, randomword
    from config import Config
    from model import SVD, RandomGuess, DeepCF

    if method == 'svd':
        config = Config('train_sub_txt', dim=100, epochs=epochs, layers=0, reg=.02, keep_prob=.9, clean=clean)
        name = 'svd_' + randomword(10)

        test = np.array(config.data.df_test)
        svd = SVD(config, model_name=name)
        acc, max_acc = svd.train()
        print 'final acc', acc, 'max acc', max_acc
        rate = svd.predict(test)
        final_acc = my_acc(rate, test[:, 2])
        print 'mse', my_rmse(rate, test[:, 2]), 'final test acc', final_acc

        all_test = config.data.get_all_test()
        rate = svd.predict(all_test)
        array = config.data.get_origin_array(rate)

        config.data.save_res(array, name=name)
        assert not queue.full()
        queue.put((name, final_acc))
    else:
        config = Config('train_sub_txt', dim=100, epochs=epochs, layers=4, reg=.02, keep_prob=.9, clean=clean)
        _, test = config.data.make_batch()
        name = 'deep_cf_' + randomword(10)

        deep_cf = DeepCF(config, model_name=name)
        acc, max_acc = deep_cf.train()
        print 'final acc', acc, 'max acc', max_acc

        rate = deep_cf.predict(test)
        rate_gt = np.argmax(test[2], axis=1) + 1
        final_acc = my_acc(rate, rate_gt)
        print 'rmse', my_rmse(rate, rate_gt), 'final test acc', final_acc

        rate = deep_cf.predict(config.data.make_all_test_batch())
        array2 = config.data.get_origin_array(rate)

        config.data.save_res(array2, name=name)

        queue.put((name, final_acc))
    return 0


epochs = 5000
queue = multiprocessing.Queue(100)
parallel, clean = True, True
tasks = []
one_flag = False
for _ind in range(5):
    method = 'svd'
    if parallel:
        if not one_flag:
            t = multiprocessing.Process(target=client, args=(queue, method, False, epochs))
            one_flag = True
        else:
            t = multiprocessing.Process(target=client, args=(queue, method, clean, epochs))
        t.start()
        tasks.append(t)
    else:
        client(queue, method, clean, epochs)
for _ind in range(4):
    method = 'deep_cf'
    if parallel:
        t = multiprocessing.Process(target=client, args=(queue, method, clean, epochs // 30))
        t.start()
        tasks.append(t)
    else:
        client(queue, method, clean, epochs)

if parallel:
    for t in tasks:
        t.join()

arrs = []
while not queue.empty():
    arrs.append(queue.get())

import cPickle, glob
from data import Data

# with open('res.pkl', 'w')as f:
#     cPickle.dump(arrs, f,protocol=cPickle.HIGHEST_PROTOCOL)
print arrs

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
np.savetxt('arr.txt', final_arr, fmt='%.3f')
Data('train_sub_txt', clean=False, shuffle=False).save_res(final_arr, '../res', type='table')
