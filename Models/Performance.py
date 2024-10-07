import numpy as np
import torch
import copy
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
model_name = 'Transformer'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
dropout = 0.5                                              # 随机失活
require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
num_classes = 4                                     # 类别数
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
        self.Separ = SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=strides, padding=1, bias=False)
        self.Separ2 = SeparableConv2d(out_filters, out_filters, kernel_size=3, stride=strides, padding=1, bias=False)
        self.Bn = nn.BatchNorm2d(out_filters)


    def forward(self, inp):
        x = self.Separ(inp)
        x = self.Bn(x)
        x = self.relu(x)
        x = self.Separ2(x)
        x = self.Bn(x)
        x = self.relu(x)

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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.xception = Xception()
        self.transformer = Shift_Transformer()
        self.fc = nn.Linear(260, num_classes)

    def forward(self, x):
        xception_out = self.xception(x)
        #transformer_out = self.transformer(xception_out)
        transformer_out = self.transformer(x)
        concatenated_out = torch.cat((xception_out.view(xception_out.size(0), -1), transformer_out.view(transformer_out.size(0), -1)), dim=1)
        out = self.fc(concatenated_out)
        return out
# 定义模型
model = Model()  # 替换为你的模型类或实例化的对象

# 移动模型到适当的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置输入大小
input_size = (64, 256, 35)

# 打印模型摘要信息，包括参数数量和输出形状
summary(model, input_size=input_size)

# 计算 FLOPs（需要安装thop库）
from thop import profile

inputs = torch.randn(*input_size).to(device)
flops, _ = profile(model, inputs=(inputs,))

# 计算 FPS
num_iterations = 100  # 运行模型的迭代次数
total_time = 0.0

for _ in range(num_iterations):
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    _ = model(inputs)

    end_time.record()
    torch.cuda.synchronize()
    iteration_time = start_time.elapsed_time(end_time) / 1000  # 转换为秒
    total_time += iteration_time

average_fps = num_iterations / total_time

# 输出结果
print("Parameters: ", sum(p.numel() for p in model.parameters()) / 1_000, "k")
print("FLOPs: ", flops / 1_000_000, "M")
print("FPS: ", average_fps)