import random
import numpy as np
import os
import torch
from multiprocessing import Pool, Manager
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
from tqdm import tqdm

# 用于写入pickle的方法
def pickle_write(file_path, imgs):
    f = open(file_path, 'wb')
    pickle.dump(imgs, f)
    f.close()

def compute_distance(i, data1, distance1, queue):
    distances = []
    for j in range(len(data1)):
        if j <= i: # 避免重复计算已经计算过的距离
            distances.append(distance1[i][j])
        else:
            np1=np.transpose(data1[i], (1, 0))
            np2=np.transpose(data1[j], (1, 0))
            alldistance, way = fastdtw(np1, np2, dist=euclidean)
            distances.append(alldistance)
    queue.put((i, distances))
def data_processing(data0,distance0):
    num_processes = 6  # 进程数
    pool = Pool(processes=num_processes)  # 创建进程池
    results = []  # 存储结果
    queue = manager.Queue()  # 创建队列，用于存储每个进程计算的结果
    for i in range(len(data0)):
        pool.apply_async(compute_distance, args=(i, data0, distance0, queue,))
    pbar = tqdm(total=len(data0), desc="计算距离", unit="tensor")
    for i in range(len(data0)):
        result = queue.get()  # 获取进程计算的结果
        results.append(result[1])
        distance0[result[0]] = result[1]
        pbar.update(1)
    pbar.close()
    pool.close()
    pool.join()  # 等待所有进程结束
    return data0,distance0

if __name__ == '__main__':
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

    for k in range(len(datalist)):
        d1path = datalist.pop()
        l1path = labellist.pop()
        p1 = path + d1path
        p2 = path + l1path
        print("读取数据文件：" + d1path)
        data = np.loadtxt(p1, delimiter=",", skiprows=0)
        datatensor = torch.tensor(data)  # 将整一个数据文件转换为一个大tensor
        datatensors = datatensor.split(256, 0)  # 将这个大tensor按行256长度进行切分

        label = np.loadtxt(p2, delimiter="	", skiprows=1)
        labeltensors = torch.tensor(label)

        data0 = []  # 标签为3的数据集
        data1 = []  # 标签为4的数据集
        for l in range(labeltensors.shape[0]):
            if (int(labeltensors[l]) == 1):
                data0.append(datatensors[l].float())
            if (int(labeltensors[l]) == 2):
                data1.append(datatensors[l].float())


        manager = Manager()  # 创建线程安全的数据管理器
        distance0 = manager.list(np.zeros((len(data0), len(data0))))  # 注册线程安全的二维数组
        distance1 = manager.list(np.zeros((len(data1), len(data1))))  # 注册线程安全的二维数组


        data0, distance0=data_processing(data0,distance0)
        data1, distance1 = data_processing(data1, distance1)


        distance0 = np.asarray(distance0)
        distance1 = np.asarray(distance1)

        pickle_write('column_data0'+d1path + '.pickle', data0)
        pickle_write('column_data1'+d1path + '.pickle', data1)

        pickle_write('column_distance0' +d1path + '.pickle', distance0)
        pickle_write('column_distance1' +d1path + '.pickle', distance1)




