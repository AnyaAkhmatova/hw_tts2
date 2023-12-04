import torch
import torch.nn as nn

from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, dilation):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv1d(in_channels=n_channels, 
                        out_channels=n_channels, 
                        kernel_size=kernel_size, 
                        padding=(kernel_size-1)*dilation[0][0]//2, 
                        dilation=dilation[0][0], 
                        bias=True)),
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv1d(in_channels=n_channels, 
                        out_channels=n_channels, 
                        kernel_size=kernel_size, 
                        padding=(kernel_size-1)*dilation[0][1]//2, 
                        dilation=dilation[0][1], 
                        bias=True))
        )
        self.block2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv1d(in_channels=n_channels, 
                        out_channels=n_channels, 
                        kernel_size=kernel_size, 
                        padding=(kernel_size-1)*dilation[1][0]//2, 
                        dilation=dilation[1][0], 
                        bias=True)),
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv1d(in_channels=n_channels, 
                        out_channels=n_channels, 
                        kernel_size=kernel_size, 
                        padding=(kernel_size-1)*dilation[1][1]//2, 
                        dilation=dilation[1][1], 
                        bias=True))
        )
        self.block3 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv1d(in_channels=n_channels, 
                        out_channels=n_channels, 
                        kernel_size=kernel_size, 
                        padding=(kernel_size-1)*dilation[2][0]//2, 
                        dilation=dilation[2][0], 
                        bias=True)),
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv1d(in_channels=n_channels, 
                        out_channels=n_channels, 
                        kernel_size=kernel_size, 
                        padding=(kernel_size-1)*dilation[2][1]//2, 
                        dilation=dilation[2][1], 
                        bias=True))
        )
        
    def forward(self, x):
        y = self.block1(x) + x
        y = self.block2(y) + y
        y = self.block3(y) + y
        return y


class MRF(nn.Module):
    def __init__(self, n_channels, kernel_sizes, dilation):
        super().__init__()
        self.layers = nn.ModuleList([ResBlock(n_channels, kernel_size, dilation) \
                                     for kernel_size in kernel_sizes])
        
    def forward(self, x):
        y = torch.zeros_like(x).to(x.device)
        for layer in self.layers:
            y += layer(x)
        y /= len(self.layers)
        return y


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_tr_ks, conv_tr_stride, kernel_sizes, dilation):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.upsampling = weight_norm(nn.ConvTranspose1d(in_channels, 
                                                         out_channels, 
                                                         conv_tr_ks, 
                                                         stride=conv_tr_stride,
                                                         padding=(conv_tr_ks-conv_tr_stride)//2))
        self.mrf = MRF(out_channels, kernel_sizes, dilation)
       
    def forward(self, x):
        y = self.leaky_relu(x)
        y = self.upsampling(y)
        y = self.mrf(y)
        return y
    

class Generator(nn.Module):
    def __init__(self, n_blocks, start_channels, conv_tr_ks, kernel_sizes, dilation):
        super().__init__()

        self.conv_begin = weight_norm(nn.Conv1d(80, start_channels,
                                                kernel_size=7, padding=3))

        blocks = []
        for i in range(n_blocks):
            blocks.append(
                GeneratorBlock(start_channels // (2**i), 
                               start_channels // (2**(i+1)),
                               conv_tr_ks[i],
                               conv_tr_ks[i]//2,
                               kernel_sizes,
                               dilation)
            )
        self.blocks = nn.ModuleList(blocks)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv_end = weight_norm(nn.Conv1d(start_channels // (2**n_blocks), 1, 
                                              kernel_size=7, padding=3))
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        y = self.conv_begin(x)
        for block in self.blocks:
            y = block(y)
        y = self.leaky_relu(y)
        y = self.conv_end(y)
        y = self.tanh(y)
        return y

