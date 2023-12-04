import torch.nn as nn

from torch.nn.utils import weight_norm


class MPDBlock(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)))
        ])

        self.conv_end1 = weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), padding=(2, 0)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv_end2 = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(2, 0)))

    def forward(self, x):
        if x.shape[2] % self.p != 0:
            y = nn.functional.pad(x, (0, self.p - x.shape[2] % self.p))
        else:
            y = x
        y = y.reshape(y.shape[0], 1, -1, self.p)
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
    

class MPD(nn.Module):
    def __init__(self, ps):
        super().__init__()
        self.ps = ps

        self.blocks = nn.ModuleList([MPDBlock(p) for p in ps])

    def forward(self, x, x_target):
        y = x.reshape(x.shape[0], 1, -1)
        y_target = x_target.reshape(x_target.shape[0], 1, -1)

        y_res = []
        y_target_res = []
        fmaps_res = []
        fmaps_target_res = []

        for block in self.blocks:
            cur_y, cur_fmaps = block(y)
            cur_y_target, cur_fmaps_target = block(y_target)
            y_res += [cur_y]
            y_target_res += [cur_y_target]
            fmaps_res += [cur_fmaps]
            fmaps_target_res += [cur_fmaps_target]

        return y_res, fmaps_res, y_target_res, fmaps_target_res
    
