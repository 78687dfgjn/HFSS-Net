import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowStd(nn.Module):






    def __init__(self, kernel_size=3, channels=None, eps=1e-5):







        super(WindowStd, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2, "kernel_size必须是整数或二元组"
            self.kernel_size = kernel_size

        self.channels = channels
        self.eps = eps
        self.weight = None


        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


        if self.channels is not None:
            self._init_weight()

    def _init_weight(self):

        kernel_h, kernel_w = self.kernel_size
        kernel_area = kernel_h * kernel_w

        single_kernel = torch.ones(1, 1, kernel_h, kernel_w) / kernel_area

        self.weight = single_kernel.repeat(self.channels, 1, 1, 1)

        self.register_buffer('mean_kernel', self.weight)

    def forward(self, x):














        assert len(x.shape) == 4, "输入必须是4维张量 (B, C, H, W)"
        batch_size, channels, height, width = x.shape


        if self.channels is None:
            self.channels = channels
            self._init_weight()
        else:
            assert self.channels == channels, f"输入通道数({channels})与初始化通道数({self.channels})不匹配"


        x_padded = F.pad(x,
                        pad=(self.padding[1], self.padding[1],
                             self.padding[0], self.padding[0]),
                        mode='reflect')


        mean = F.conv2d(x_padded,
                       weight=self.mean_kernel,
                       bias=None,
                       stride=1,
                       padding=0,
                       groups=channels)


        x_squared = x **2
        x_squared_padded = F.pad(x_squared,
                                pad=(self.padding[1], self.padding[1],
                                     self.padding[0], self.padding[0]),
                                mode='reflect')
        mean_squared = F.conv2d(x_squared_padded,
                               weight=self.mean_kernel,
                               bias=None,
                               stride=1,
                               padding=0,
                               groups=channels)


        std = torch.sqrt(torch.clamp(mean_squared - mean** 2, min=self.eps))

        return std

class SDGW(nn.Module):
    def __init__(self, ch):
        super(SDGW, self).__init__()

        self.convl2l = nn.Conv2d(ch, ch , 3, 1, 1)
        self.conv1 = nn.Conv2d(ch, ch, 1, bias=False)
        self.std = WindowStd(3, ch)
        self.sig = nn.Sigmoid()



    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)

        feature = self.convl2l(x_l)
        std = self.std(x_l)
        weight = self.sig(self.conv1(std))
        out_xl = feature * weight

        return out_xl
