
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowStd(nn.Module):
    """
    实现窗口标准差统计的模块
    输入: (B, C, H, W) 形状的特征图
    输出: (B, C, H, W) 形状的特征图，与输入尺寸一致
    使用镜像padding保证边缘计算的准确性
    """
    def __init__(self, kernel_size=3, channels=None, eps=1e-5):
        """
        参数:
            kernel_size (int or tuple): 窗口大小，默认3x3
            channels (int): 输入特征图的通道数，用于初始化卷积核
                            若为None，需在第一次前向传播时自动推断
            eps (float): 数值稳定性参数，避免除以零或开方负数
        """
        super(WindowStd, self).__init__()
        # 统一kernel_size为元组格式
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2, "kernel_size必须是整数或二元组"                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
            self.kernel_size = kernel_size

        self.channels = channels
        self.eps = eps
        self.weight = None  # 用于存储均值卷积核（计算窗口内均值）

        # 计算padding大小（保证输出尺寸与输入一致）
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!

        # 初始化卷积核（如果通道数已知）
        if self.channels is not None:
            self._init_weight()

    def _init_weight(self):
        """初始化均值卷积核：每个通道的窗口内权重均为1/(k_h*k_w)"""
        kernel_h, kernel_w = self.kernel_size
        kernel_area = kernel_h * kernel_w
        # 创建单个通道的卷积核 (1, 1, k_h, k_w)
        single_kernel = torch.ones(1, 1, kernel_h, kernel_w) / kernel_area
        # 扩展到所有通道 (C, 1, k_h, k_w)，每个通道独立计算
        self.weight = single_kernel.repeat(self.channels, 1, 1, 1)
        # 将权重注册为非参数化缓冲区（不参与训练）
        self.register_buffer('mean_kernel', self.weight)

    def forward(self, x):
        """
        前向传播：对输入特征图应用窗口标准差计算

        标准差计算原理：
        std = sqrt( E[x²] - (E[x])² )
        其中E[·]表示窗口内的均值

        参数:
            x: 输入特征图，形状为 (B, C, H, W)

        返回:
            窗口标准差特征图，形状为 (B, C, H, W)
        """
        # 检查输入形状
        assert len(x.shape) == 4, "输入必须是4维张量 (B, C, H, W)"
        batch_size, channels, height, width = x.shape

        # 如果通道数未初始化，自动推断并初始化卷积核
        if self.channels is None:
            self.channels = channels
            self._init_weight()
        else:
            assert self.channels == channels, f"输入通道数({channels})与初始化通道数({self.channels})不匹配"                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!

        # 镜像padding
        x_padded = F.pad(x,
                        pad=(self.padding[1], self.padding[1],  # 左右padding                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
                             self.padding[0], self.padding[0]),  # 上下padding
                        mode='reflect')

        # 计算窗口内均值 E[x]
        mean = F.conv2d(x_padded,
                       weight=self.mean_kernel,
                       bias=None,
                       stride=1,
                       padding=0,
                       groups=channels)

        # 计算窗口内平方的均值 E[x²]
        x_squared = x **2
        x_squared_padded = F.pad(x_squared,
                                pad=(self.padding[1], self.padding[1],                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
                                     self.padding[0], self.padding[0]),                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
                                mode='reflect')
        mean_squared = F.conv2d(x_squared_padded,
                               weight=self.mean_kernel,
                               bias=None,
                               stride=1,
                               padding=0,
                               groups=channels)

        # 计算标准差：sqrt(E[x²] - (E[x])² + eps)
        std = torch.sqrt(torch.clamp(mean_squared - mean** 2, min=self.eps))                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!

        return std

class SDGW(nn.Module):
    def __init__(self, ch):
        super(SDGW, self).__init__()

        self.convl2l = nn.Conv2d(ch, ch , 3, 1, 1)
        self.conv1 = nn.Conv2d(ch, ch, 1, bias=False)
        self.std = WindowStd(3, ch)
        self.sig = nn.Sigmoid()
        # self.convl2g = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        # self.convg2l = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        #
        feature = self.convl2l(x_l)
        std = self.std(x_l)
        weight = self.sig(self.conv1(std))
        out_xl = feature * weight

        return out_xl

# 使用示例
