from math import log

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from DataLoader.LoadData_2C import Dataset_train, Dataset_test
import torch.nn.functional as F
import numpy as np
import copy
#数据可视化相关
from tensorboardX import SummaryWriter
logger = SummaryWriter(log_dir="../../data/log")
model_name = 'Transformer'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
dropout = 0.5                                              # 随机失活
require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
num_classes = 2                                     # 类别数
n_vocab = 0                                                # 词表大小，在运行时赋值
num_epochs = 2000                                          # epoch数
batch_size = 64                                          # mini-batch大小
pad_size = 256                                             # 每句话处理成的长度(短填长切)
learning_rate = 5e-4                                       # 学习率
embed = 35                                                 # 字向量维度
dim_model = 35
hidden = 256
last_hidden = 256
num_head = 5
num_encoder = 1
lam=1/12


#ShiftVit模块
def shift_feat(x, n_div):
    x = torch.reshape(x, (-1, 35, 16, 16))
    B, C, H, W = x.shape
    g = C // n_div
    out = torch.zeros_like(x)

    out[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # shift left
    out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
    out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
    out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down

    out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
    return out


class Shift_Transformer(nn.Module):
    def __init__(self):
        super(Shift_Transformer, self).__init__()
        self.postion_embedding = Positional_Encoding(embed, pad_size, dropout, device)
        # 减少编码器数量
        self.encoders = nn.ModuleList([
            copy.deepcopy(Encoder(dim_model, num_head, hidden, dropout))
            for _ in range(num_encoder // 2)])
        self.fc1 = nn.Linear(pad_size * dim_model, num_classes)

    def forward(self, x):
        out = torch.reshape(x, (-1, 256, 35))
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)
    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out
class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        context = shift_feat(x, lam)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class eca_layer(nn.Module):
    def __init__(self, channel=35, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 9))
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        #torch.Size([256, 35, 1, 256])
        y = self.avg_pool(x)
        #torch.Size([256, 35, 1, 9])
        y = self.conv(y.squeeze())
        #torch.Size([256, 35, 7])
        y = self.conv(y.squeeze())
        # torch.Size([256, 35, 5])
        y = self.conv(y.squeeze())
        # torch.Size([256, 35, 3])
        y = self.conv(y.squeeze())
        # torch.Size([256, 35, 1])
        y = self.sigmoid(y)
        y=y.unsqueeze(-1)
        x = x * y.expand_as(x)
        #torch.Size([256, 35, 1, 256])
        return x


# 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # 逐通道卷积：groups=in_channels=out_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        # 逐点卷积：普通1x1卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True):
        super(Block, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        rep = []

        for i in range(reps):
            if i == 0 and strides != 1:
                rep.append(
                    SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=strides, padding=1, bias=False))
            else:
                rep.append(SeparableConv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False))

            rep.append(nn.BatchNorm2d(out_filters))
            if i < reps - 1:
                rep.append(self.relu)

        self.rep = nn.Sequential(*rep)

        if in_filters != out_filters or strides != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(out_filters)
            )
        else:
            self.skip = None

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp

        x += skip
        return x
class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        self.num_classes = num_classes  # 总分类数

        self.eca_layer = eca_layer()

        ################################## 定义 Entry flow ###############################################################

        # do relu here
        # Block中的参数顺序：in_filters,out_filters,reps,stride,start_with_relu,grow_first
        self.block1 = Block(35, 64, 2, 2, start_with_relu=False)

        self.fc = nn.Linear(64, last_hidden)
        ###################################################################################################################


    def forward(self, x):
        # 转置操作，将时间步放在通道维度上方
        x = torch.reshape(x, (-1, 35, 1, 256))
        #x = torch.reshape(x, (-1, 35, 16, 16))

        x = self.eca_layer(x)
        x = torch.reshape(x, (-1, 35, 16, 16))
        ################################## 定义 Entry flow ###############################################################

        x = self.block1(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CascadeModel(nn.Module):
    def __init__(self):
        super(CascadeModel, self).__init__()
        self.xception = Xception()
        self.transformer = Shift_Transformer()
        self.fc = nn.Linear(258, num_classes)

    def forward(self, x):
        xception_out = self.xception(x)
        #transformer_out = self.transformer(xception_out)
        transformer_out = self.transformer(x)
        concatenated_out = torch.cat((xception_out.view(xception_out.size(0), -1), transformer_out.view(transformer_out.size(0), -1)), dim=1)
        out = self.fc(concatenated_out)
        return out


# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进。
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = next(net.parameters()).device

    n_classes = num_classes
    class_cnts = [0] * n_classes  # 每个类别的样本数
    tp = [0] * n_classes  # 每个类别的 True Positive 数量
    fp = [0] * n_classes  # 每个类别的 False Positive 数量
    fn = [0] * n_classes  # 每个类别的 False Negative 数量

    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_pred = net(X).argmax(dim=1)

            for i in range(n_classes):
                mask = (y == i)
                class_cnts[i] += mask.sum().item()
                tp[i] += ((y_pred == i) & mask).sum().item()
                fp[i] += ((y_pred == i) & (~mask)).sum().item()
                fn[i] += ((y_pred != i) & mask).sum().item()

    precisions = []
    recalls = []
    f1_scores = []
    for i in range(n_classes):
        if tp[i] + fp[i] == 0:
            p = 0
        else:
            p = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            r = 0
        else:
            r = tp[i] / (tp[i] + fn[i])
        if p + r == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    tp_sum = sum(tp)
    fp_sum = sum(fp)
    fn_sum = sum(fn)
    micro_precision = tp_sum / (tp_sum + fp_sum)
    micro_recall = tp_sum / (tp_sum + fn_sum)
    micro_f1_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    macro_precision = sum(precisions) / n_classes
    macro_recall = sum(recalls) / n_classes
    macro_f1_score = sum(f1_scores) / n_classes

    # 用字典记录6个指标，并返回
    metrics = {'micro_p': micro_precision,
               'micro_r': micro_recall,
               'micro_f1': micro_f1_score,
               'macro_p': macro_precision,
               'macro_r': macro_recall,
               'macro_f1': macro_f1_score}
    return metrics


# 本函数已保存在d2lzh_pytorch包中方便以后使用
import csv
import time
import torch


def train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, erp, erp_type):
    net = net.to(device)
    print("training on", device)
    loss_fn = torch.nn.CrossEntropyLoss()

    max_micro_p = max_micro_r = max_micro_f1 = 0
    max_macro_p = max_macro_r = max_macro_f1 = 0

    best_model_state_dict = None  # Initialize variable for storing the best model state dict
    best_macro_f1 = 0  # Initialize variable for tracking the best macro F1-score

    for epoch in range(num_epochs):
        train_l_sum, n, batch_count, start = 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            n += y.shape[0]
            batch_count += 1

        test_metrics = evaluate_accuracy(test_iter, net)

        max_micro_p = max(max_micro_p, test_metrics['micro_p'])
        max_micro_r = max(max_micro_r, test_metrics['micro_r'])
        max_micro_f1 = max(max_micro_f1, test_metrics['micro_f1'])

        max_macro_p = max(max_macro_p, test_metrics['macro_p'])
        max_macro_r = max(max_macro_r, test_metrics['macro_r'])
        max_macro_f1 = max(max_macro_f1, test_metrics['macro_f1'])

        if max_macro_f1 > best_macro_f1:  # Check if current macro F1-score is better than the previous best
            best_macro_f1 = max_macro_f1
            best_model_state_dict = net.state_dict()  # Save the state dict of the current best model

        epoch_loss = train_l_sum / batch_count

        print('epoch:', epoch + 1, 'loss:', epoch_loss)
        print('max micro-precision:', max_micro_p, end=',')
        print('max micro-recall:', max_micro_r, end=',')
        print('max micro F1-score:', max_micro_f1, end=',')
        print('max macro-precision:', max_macro_p, end=',')
        print('max macro-recall:', max_macro_r, end=',')
        print('max macro F1-score:', max_macro_f1)

        with open(erp + "_" + erp_type + ".csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # 如果文件为空，写入表头
                writer.writerow(
                    ['epoch', 'loss', 'micro_p', 'micro_r', 'micro_f1', 'macro_p', 'macro_r', 'macro_f1', 'max_micro_p',
                     'max_micro_r', 'max_micro_f1', 'max_macro_p', 'max_macro_r', 'max_macro_f1'])
            writer.writerow(
                [epoch + 1, epoch_loss, test_metrics['micro_p'], test_metrics['micro_r'], test_metrics['micro_f1'],
                 test_metrics['macro_p'], test_metrics['macro_r'], test_metrics['macro_f1'], max_micro_p, max_micro_r,
                 max_micro_f1, max_macro_p, max_macro_r, max_macro_f1])

    if best_model_state_dict is not None:  # Save the state dict of the best model to a file
        torch.save(best_model_state_dict, erp+"_best_model.pt")

erp="N400"
erp_type="bini"
batch_size = 64
net = CascadeModel().to(device)
train_data = Dataset_train(transform=transforms.ToTensor(),
                               allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k2\\")
test_data = Dataset_test(transform=transforms.ToTensor(),
                             allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k2\\")
train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
lr, num_epochs = 0.01, 100
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, erp, erp_type)
"""

erp="P3"
erp_type="bini"
batch_size = 64
net = CascadeModel().to(device)
train_data = Dataset_train(transform=transforms.ToTensor(),
                               allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
test_data = Dataset_test(transform=transforms.ToTensor(),
                             allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
lr, num_epochs = 0.01, 100
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, erp, erp_type)

erp="N2PC"
erp_type="bini"
batch_size = 64
net = CascadeModel().to(device)
train_data = Dataset_train(transform=transforms.ToTensor(),
                               allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
test_data = Dataset_test(transform=transforms.ToTensor(),
                             allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
lr, num_epochs = 0.01, 100
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, erp, erp_type)

erp="LRP"
erp_type="bini"
batch_size = 64
net = CascadeModel().to(device)
train_data = Dataset_train(transform=transforms.ToTensor(),
                               allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
test_data = Dataset_test(transform=transforms.ToTensor(),
                             allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
lr, num_epochs = 0.01, 100
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, erp, erp_type)



erp="N170"
erp_type="bini"
batch_size = 64
net = CascadeModel().to(device)
train_data = Dataset_train(transform=transforms.ToTensor(),
                               allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
test_data = Dataset_test(transform=transforms.ToTensor(),
                             allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
lr, num_epochs = 0.01, 100
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, erp, erp_type)

erp="ERN"
erp_type="binlabel"
batch_size = 64
net = CascadeModel().to(device)
train_data = Dataset_train(transform=transforms.ToTensor(),
                               allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
test_data = Dataset_test(transform=transforms.ToTensor(),
                             allpath="F:\\研究生文件\\小论文2相关资料\\开源代码\\第二篇小论文+大论文相关内容\\DataProgress\\输出平均数据_gpt优化\\" + erp + "输出数据\\" + erp_type + "\\按列为时间步\\k1\\")
train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
lr, num_epochs = 0.01, 100
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, erp, erp_type)

"""
