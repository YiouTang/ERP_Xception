import os
import pickle
import numpy as np
import torch

# 读取pickle
def pickle_read(file_path):
    f = open(file_path, 'rb')
    data = pickle.load(f)
    return data
# 写入pickle
def pickle_write(file_path, imgs):
    f = open(file_path, 'wb')
    pickle.dump(imgs, f)
    f.close()

allpath="F:\\研究生文件\\小论文2相关资料\\实验数据\\N400\\biniLabel\\"

path = allpath
dirs = os.listdir(path)  # 获取指定路径下的文件
# print(dirs)
datalist = []
labellist = []
for i in dirs:  # 循环读取路径下的文件并筛选输出
    if os.path.splitext(i)[1] == ".CSV":  # 筛选npz文件
        datalist.append(i)
    if os.path.splitext(i)[1] == ".csv":  # 筛选npz文件
        labellist.append(i)

for i in range(len(datalist)):
    imgs = []
    getdata=path + datalist.pop()
    getlabel=path + labellist.pop()
    print(getdata)
    print(getlabel)
    data = np.loadtxt(getdata, delimiter=",", skiprows=0)
    datatensor = torch.tensor(data)
    datatensors = datatensor.split(256, 0)
    label = np.loadtxt(getlabel, delimiter="	", skiprows=1)
    labeltensors = torch.tensor(label)
    for j in range(labeltensors.shape[0]):
        lab = int(labeltensors[j])
        imgs.append((datatensors[j].float(), lab))
    pickle_write("N400_biniLabel_"+str(i+1)+".pickle",imgs)