import torch
import torch.nn as nn
from torch.nn import functional as F
from models.modules.CSAG import SABlock, conv3x3
class Dsv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(Dsv, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )
    def forward(self, input):
        return self.dsv(input)
class ScaleAttentionBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, use_cbam=True, no_spatial=False, drop_out=False):
        super(ScaleAttentionBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.dropout = drop_out

        self.relu = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(in_size, out_size)
        self.bn3 = nn.BatchNorm2d(out_size)
        if use_cbam:
            self.cbam = SABlock(in_size, reduction_ratio=4, no_spatial=self.no_spatial)  # out_size
        else:
            self.cbam = None
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(x)
        out += residual
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)
        return out