import os, subprocess
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from utils import root_path, my_acc, my_rmse,randomword
from config import Config
from model import SVD, RandomGuess, DeepCF

os.chdir(root_path)
subprocess.call('./clean.sh')

clean = True
epochs = 6000

# svd

config = Config('train_sub_txt', dim=100, epochs=epochs, layers=0, reg=.02, keep_prob=.9, clean=clean)
test = np.array(config.data.df_test)

svd = SVD(config,'svd_'+randomword(10))
acc, max_acc = svd.train()
print 'final acc', acc, 'max acc', max_acc
rate = svd.predict(test)
print 'mse', my_rmse(rate, test[:, 2]), 'final test acc', my_acc(rate, test[:, 2])

all_test = config.data.get_all_test()
rate = svd.predict(all_test)
array = config.data.get_origin_array(rate)
config.data.save_res(array, name='svd' + str(clean))

# daul net

config = Config('train_sub_txt', dim=100, epochs=epochs, layers=4, reg=.02, keep_prob=.9, clean=clean)
_, test = config.data.make_batch()
deep_cf = DeepCF(config,'deep_cf_'+randomword(10))
acc, max_acc = deep_cf.train()
print 'final acc', acc, 'max acc', max_acc

rate = deep_cf.predict(test)
rate_gt = np.argmax(test[2], axis=1) + 1
print 'rmse', my_rmse(rate, rate_gt), 'final test acc', my_acc(rate, rate_gt)

rate = deep_cf.predict(config.data.make_all_test_batch())
array2 = config.data.get_origin_array(rate)
config.data.save_res(array2, name='deep_cf' + str(clean))
