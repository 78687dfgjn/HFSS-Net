import torch
import torch.nn as nn
import torch.nn.functional as F


class PFADWT(nn.Module):
    def __init__(self, channels):
        super().__init__()

        filters = torch.tensor([
            [[1.,  1.],
             [1.,  1.]],

            [[-1., -1.],
             [ 1.,  1.]],

            [[-1.,  1.],
             [-1.,  1.]],

            [[ 1., -1.],
             [-1.,  1.]]
        ]) / 2.0

        self.channels = channels
        self.register_buffer(
            "dwt_filters",
            filters.unsqueeze(1).repeat(channels, 1, 1, 1)
        )

        self.conv1 = nn.Conv2d(channels * 4, channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dconv = nn.ConvTranspose2d(
            channels, channels, kernel_size=2, stride=2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.shape

        ph = h % 2
        pw = w % 2

        x_pad = F.pad(x, (0, pw, 0, ph)) if ph or pw else x

        f = F.conv2d(
            x_pad,
            self.dwt_filters.to(dtype=x.dtype),
            stride=2,
            groups=self.channels
        )

        f = self.relu(self.conv1(f))
        m = self.sigmoid(self.dconv(f))
        m = m[:, :, :h, :w]

        return x * m + x
