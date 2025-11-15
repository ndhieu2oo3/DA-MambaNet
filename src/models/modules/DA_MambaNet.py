import torch
import torch.nn as nn
from models.modules.SelectiveKernels import SKConv_7, SKUnit
from models.modules.MultiAttentionMamba import MultiAttentionMamba
from models.modules.CSAG import CSAG
from models.modules.AxialResidualBlock import AxialResidualBlock
from models.modules.ScaleAttention import Dsv, ScaleAttentionBlock

class DA_MambaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pw_in = nn.Conv2d(3, 16, kernel_size=1)
        self.sk_in = SKConv_7(16, M=2, G=16, r=4, stride=1 ,L=32)
        """Encoder"""
        self.e1 = MultiAttentionMamba(16, 32, 256, 192)
        self.e2 = MultiAttentionMamba(32, 64, 128, 96)
        self.e3 = MultiAttentionMamba(64, 128, 64, 48)
        self.e4 = MultiAttentionMamba(128, 256, 32, 24)
        self.e5 = MultiAttentionMamba(256, 512, 16, 12)
        """AttentionGate"""
        self.a1 = CSAG(32,32,16)
        self.a2 = CSAG(64,64,32)
        self.a3 = CSAG(128,128,64)
        self.a4 = CSAG(256,256,128)
        self.a5 = CSAG(512,512,256)
        """Bottle Neck"""
        self.b5 = SKUnit(512, 512, 512, M=2, G=16, r=2, stride=1, L=32)
        """Decoder"""
        self.d5 = AxialResidualBlock(512+512,256)
        self.d4 = AxialResidualBlock(256+256,128)
        self.d3 = AxialResidualBlock(128+128,64)
        self.d2 = AxialResidualBlock(64+64,32)
        self.d1 = AxialResidualBlock(32+32,16)
        # deep supervision
        self.dsv5 = Dsv(256, 4, scale_factor=(192,256))
        self.dsv4 = Dsv(128,4, scale_factor=(192,256))
        self.dsv3 = Dsv(64, 4, scale_factor=(192,256))
        self.dsv2 = Dsv(32, 4, scale_factor=(192,256))
        self.dsv1 = Dsv(16, 4, scale_factor=(192,256))
        self.scale_attention = ScaleAttentionBlock(16,4)
        self.conv_final = nn.Conv2d(16, 1, kernel_size=1)
        self.conv_out = nn.Conv2d(4, 1, kernel_size=1)
    def forward(self, x):
        """Encoder"""
        x = self.pw_in(x)
        x = self.sk_in(x)

        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)
        """BottleNeck"""
        x = self.b5(x)
        """Skip connection"""
        x5 = self.a5(x, skip5)
        x5 = self.d5(x5)
        x4 = self.a4(x5, skip4)
        x4 = self.d4(x4)
        x3 = self.a3(x4, skip3)
        x3 = self.d3(x3)
        x2 = self.a2(x3, skip2)
        x2 = self.d2(x2)
        x1 = self.a1(x2, skip1)
        x1 = self.d1(x1)
        decoder_out = self.conv_final(x1)
        "Deep Supervision"
        dsv5 = self.dsv5(x5)
        dsv4 = self.dsv4(x4)
        dsv3 = self.dsv3(x3)
        dsv2 = self.dsv2(x2)
        dsv1 = self.dsv1(x1)
        dsv = torch.cat([dsv5, dsv4, dsv3, dsv2], dim=1)
        dsv = self.scale_attention(dsv)
        layer_out = self.conv_out(dsv)
        final_out = decoder_out + layer_out
        return final_out, decoder_out, layer_out