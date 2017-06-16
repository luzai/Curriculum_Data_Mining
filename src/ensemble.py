import os

import matplotlib

matplotlib.use("TkAgg")

from utils import root_path

os.chdir(root_path)
from config import Config
from svd import SVD

# subprocess.call('./clean.sh')
config = Config('train_sub_txt', dim=100, epochs=5000, layers=0, reg=.02, keep_prob=.85, clean=True)
# config = Config('ml-20m', dim=5, epochs=5000,layers=3,reg=.02,keep_prob=.5)
# config = Config('ml-1m', dim=5, epochs=5000, layers=0, reg=.9, keep_prob=.1)
svd = SVD(config)
svd.train()
if config.name == 'train_sub_txt':
    rate = svd.predict(config.data.get_all_test())
    # utils.my_plot(rate)
    # print rate
    array = config.data.get_origin_array(rate)
    print array
    config.data.save_res(array)
