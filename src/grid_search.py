import os
import os.path as osp

import matplotlib

matplotlib.use("TkAgg")
import multiprocessing

root_path = root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)
os.chdir(root_path)

def train(queue, dim):
    sucess=False
    while not sucess:
        try:
            from config import Config
            from svd import SVD
            config = Config('train_sub_txt', dim=dim, epochs=1000, layers=0, reg=.02, keep_prob=.9)
            svd = SVD(config)
            acc = svd.train()
            print dim,acc
            queue.put((dim, acc))
            sucess=True
        except Exception as inst :
            print dim,inst
            # queue.put((dim,str(inst)))
            # exit(-100)
            sucess=False



# subprocess.call('./clean.sh')
queue = multiprocessing.Queue(60)

# config = Config('train_sub_txt', dim=100, epochs=5000, layers=0, reg=.02, keep_prob=.5)
# print config.nb_users,config.dim,config

# config = Config('ml-20m', dim=5, epochs=5000,layers=3,reg=.02,keep_prob=.5)
# config = Config('ml-1m', dim=5, epochs=5000, layers=0, reg=.9, keep_prob=.1)
res = []
tasks = []
for dim in range(97,102):
    for _ in range(10):
        task = multiprocessing.Process(target=train, args=(queue, dim))
        tasks.append(task)
        task.start()
    import time

    time.sleep(1)
    if len(tasks) == 20:
        while tasks:
            tasks.pop().join()
        while not queue.empty():
            res.append(queue.get())
        print res

while tasks:
    tasks.pop().join()
while not queue.empty():
    res.append(queue.get())
print res
import numpy as np
a = np.array(res)
print a
from utils import my_plot
my_plot(a[:,0],a[:,1])
