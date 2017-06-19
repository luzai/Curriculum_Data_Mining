import os, time
import os.path as osp
import matplotlib
import numpy as np
from utils import my_plot, randomword
import multiprocessing, subprocess

matplotlib.use("TkAgg")
root_path = root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)
os.chdir(root_path)


def train(queue, dim):
    sucess = False
    while not sucess:
        try:
            from config import Config
            from model import SVD
            config = Config('train_sub_txt', dim=dim, epochs=1000, layers=0, reg=.02, keep_prob=.9)
            svd = SVD(config, model_name='svd_' + randomword(10))
            acc = svd.train()
            print dim, acc
            queue.put((dim, acc))
            sucess = True
        except Exception as inst:
            print dim, inst
            # queue.put((dim,str(inst)))
            # exit(-100)
            sucess = False


def train2(queue, size):
    sucess = False
    from config import Config
    from model import SVD, DeepCF
    while not sucess:
        try:
            config = Config('train_sub_txt', dim=100, epochs=200, layers=4, reg=.02, keep_prob=.9)
            deep_cf = DeepCF(config, model_name='deep_cf_' + randomword(10), size=size)
            acc = deep_cf.train()
            print size, acc
            queue.put((size, acc))
            sucess = True
        except Exception as inst:
            print size, inst
            sucess = False
            exit(-100)
            time.sleep(600)


subprocess.call('./clean.sh')
queue = multiprocessing.Queue(100)

res = []
tasks = []
for dim1 in range(300, 400, 99):
    dim2, dim3 = 200, 200
    # for dim2 in range(200, 300, 99):
    #     for dim3 in range(200, 300, 99):
    for _ in range(3):
        size = [dim1, dim2, dim3]
        task = multiprocessing.Process(target=train2, args=(queue, size))
        tasks.append(task)
        task.start()
        time.sleep(10)
        if len(tasks) == 7:
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

# a = np.array(res)
# print a
#
# my_plot(a[:, 0], a[:, 1])
