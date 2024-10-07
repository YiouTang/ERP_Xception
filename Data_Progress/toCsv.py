import os


def alter_input(file, old_str, new_str):
    """
    替换文件中的字符串
    :param file:文件名
    :param old_str:就字符串
    :param new_str:新字符串
    :return:
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str, new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)

def alter_name(file, add):
    # 原文件名
    n1 = file
    # 新文件名
    n2 = file+add
    # 调用改名函数，完成改名操作
    os.rename(n1, n2)


def alter_suffix(file, add):
    # 原文件名
    n1 = file
    # 新文件名
    n2 = file+add
    # 调用改名函数，完成改名操作
    os.rename(n1, n2)

def update(filePath):
    # listdir：返回指定的文件夹包含的文件或文件夹的名字的列表
    files = os.listdir(filePath)
    for file in files:
        fileName = filePath + os.sep + file
        path1 = filePath
        # 运用递归;isdir：判断某一路径是否为目录
        if os.path.isdir(fileName):
            update(fileName)
            continue
        else:
            if file.endswith('.csv'):
                test = file.replace("_origin", "")
                print("修改前:" + path1 + os.sep + file)
                print("修改后:" + path1 + os.sep + test)
                os.renames(path1 + os.sep + file, path1 + os.sep + test)


import re


def delete_line(file):
    lineList = []
    matchPattern = re.compile(r'-1')
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if matchPattern.search(line):
                pass
            else:
                lineList.append(line)
    file = open(file, 'w', encoding='UTF-8')
    for i in lineList:
        file.write(i)
    file.close()

path =  'F:\\研究生文件\\小论文2相关资料\\实验数据\\ERN\\OriginLabel'  # 设置路径
dirs = os.listdir(path)

update(path)
"""
for i in dirs:  # 循环读取路径下的文件并筛选输出
    #alter_name(path+i, ".npz")
    alter_suffix(path+i,".csv")
    #if os.path.splitext(i)[1] == ".csv":  # 筛选npz文件
    #    delete_line(path + i,)
    #    alter_input(path + i, "b\n","")
    #    alter_input(path + i, "i\n", "")
    #    alter_input(path + i, "n\n", "")
"""




# 输出结束的提示信息
print('Over'.center(20, '='))
