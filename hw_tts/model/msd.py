import torch.nn as nn

from torch.nn.utils import weight_norm, spectral_norm


class MSDBlock_spectral_norm(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7)),
            spectral_norm(nn.Conv1d(64, 128, kernel_size=41, stride=2, padding=20, groups=4)),
            spectral_norm(nn.Conv1d(128, 256, kernel_size=41, stride=2, padding=20, groups=16)),
            spectral_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20, groups=16))
        ])

        self.conv_end1 = spectral_norm(nn.Conv1d(512, 1024, kernel_size=5, stride=1, padding=2))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv_end2 = spectral_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        y = x.reshape(x.shape[0], 1, -1)
        fmaps = []

        for conv in self.convs:
            y = conv(y)
            y = self.leaky_relu(y)
            fmaps.append(y)

        y = self.conv_end1(y)
        y = self.leaky_relu(y)
        fmaps.append(y)
        y = self.conv_end2(y)

        return y, fmaps


class MSDBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7)),
            weight_norm(nn.Conv1d(64, 128, kernel_size=41, stride=2, padding=20, groups=4)),
            weight_norm(nn.Conv1d(128, 256, kernel_size=41, stride=2, padding=20, groups=16)),
            weight_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20, groups=16))
        ])

        self.conv_end1 = weight_norm(nn.Conv1d(512, 1024, kernel_size=5, stride=1, padding=2))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv_end2 = weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        y = x.reshape(x.shape[0], 1, -1)
        fmaps = []

        for conv in self.convs:
            y = conv(y)
            y = self.leaky_relu(y)
            fmaps.append(y)

        y = self.conv_end1(y)
        y = self.leaky_relu(y)
        fmaps.append(y)
        y = self.conv_end2(y)

        return y, fmaps


class MSD(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([
            MSDBlock_spectral_norm(),
            MSDBlock(),
            MSDBlock()     
        ])

        self.pool1 = nn.AvgPool1d(4, stride=2, padding=2)
        self.pool2 = nn.AvgPool1d(4, stride=2, padding=2)

    def forward(self, x, x_target):
        y = x.reshape(x.shape[0], 1, -1)
        y_target = x_target.reshape(x_target.shape[0], 1, -1)

        y_res = []
        y_target_res = []
        fmaps_res = []
        fmaps_target_res = []

        for i, block in enumerate(self.blocks):
            if i == 1:
                y = self.pool1(y)
                y_target = self.pool1(y_target)
            if i == 2:
                y = self.pool2(y)
                y_target = self.pool2(y_target)

            cur_y, cur_fmaps = block(y)
            cur_y_target, cur_fmaps_target = block(y_target)
            y_res += [cur_y]
            y_target_res += [cur_y_target]
            fmaps_res += [cur_fmaps]
            fmaps_target_res += [cur_fmaps_target]

        return y_res, fmaps_res, y_target_res, fmaps_target_res
    
