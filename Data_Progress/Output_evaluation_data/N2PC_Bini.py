import random
from multiprocessing import freeze_support

import numpy
import numpy as np
import os
import torch
import pickle
import concurrent.futures
from joblib._multiprocessing_helpers import mp
from tqdm import tqdm
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from tslearn.barycenters import softdtw_barycenter

imgs=[]#存放数据的数组
# 读取pickle
def pickle_read(file_path):
    f = open(file_path, 'rb')
    data = pickle.load(f)
    f.close()
    return data
# 写入pickle
def pickle_write(file_path, imgs):
    f = open(file_path, 'wb')
    pickle.dump(imgs, f)
    f.close()

def get_average_helper(args):
    p, datadtws06 = args
    similarToVariables = []
    for get_distance, get_data in datadtws06:
        getdata_np = get_data.numpy()
        column_p = getdata_np[:, p]
        similarToVariables.append(column_p)
    similarToVariables = np.array(similarToVariables)
    get_average = softdtw_barycenter(similarToVariables, gamma=1., max_iter=50, tol=1e-3)
    return get_average.squeeze()

def get_average(datadtws06,label):
    datadtws06_tran=[]
    for get_distance, get_data in datadtws06:
        getdata_np = get_data.numpy()
        getdata_np = np.transpose(getdata_np, (1, 0))
        datadtws06_tran.append(getdata_np)
    get_average = softdtw_barycenter(datadtws06_tran, gamma=1., max_iter=50, tol=1e-3)
    get_average = np.transpose(get_average, (1, 0))
    return (torch.tensor(get_average).float(), label)

def data_processing(data0,distance0,imgs,label):
    i = 0
    while i < len(data0):
        j = 0
        datadtws = []
        while j < len(data0):
            datadtws.append((distance0[i][j], data0[j]))
            j += 1
        # datadtws.sort(key=compare)  # 依据dtw距离进行排序
        datadtws_sorted = sorted(datadtws, key=lambda x: x[0])
        datadtws06 = datadtws_sorted[0:25]
        imgs.append(get_average(datadtws06,label))
        i += 1
    return imgs


if __name__ == '__main__':
    freeze_support()
    imgs = []
    path = "F:\\研究生文件\\小论文2相关资料\\实验数据\\LRP\\biniLabel\\"  # 设置路径
    dirs = os.listdir(path)  # 获取指定路径下的文件
    # print(dirs)
    datalist = []
    labellist = []
    for i in dirs:  # 循环读取路径下的文件并筛选输出
        if os.path.splitext(i)[1] == ".CSV":  # 筛选npz文件
            datalist.append(i)
        if os.path.splitext(i)[1] == ".csv":  # 筛选npz文件
            labellist.append(i)

    # 显示数据处理进度条
    pbar = tqdm(total=len(datalist), desc="softdba平均算法总进度", unit="数据集")
    for k in range(len(datalist)):
        d1path = datalist.pop()
        l1path = labellist.pop()
        data0 = pickle_read(
            'F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\计算距离矩阵_gpt优化\\N2PC中间数据\\按列为时间步\\bini\\column_data0' + d1path + '.pickle');
        distance0 = pickle_read(
            'F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\计算距离矩阵_gpt优化\\N2PC中间数据\\按列为时间步\\bini\\column_distance0' + d1path + '.pickle');
        data1 = pickle_read(
            'F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\计算距离矩阵_gpt优化\\N2PC中间数据\\按列为时间步\\bini\\column_data1' + d1path + '.pickle');
        distance1 = pickle_read(
            'F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\计算距离矩阵_gpt优化\\N2PC中间数据\\按列为时间步\\bini\\column_distance1' + d1path + '.pickle');

        imgs = data_processing(data0, distance0, imgs, 0)
        imgs = data_processing(data1, distance1, imgs, 1)

        pbar.update(1)  # 进度条更新
        pickle_write(
            'F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\N2PC输出数据\\bini\\按列为时间步\\' + d1path + '.pickle',
            imgs)
        imgs = []
    pbar.close()  # 关闭资源



