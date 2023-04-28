import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from .utils import Linear, Skip_Concat


class Decoder(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512]):
        super(Decoder, self).__init__()
        #   skip concat
        self.skip_concat = Skip_Concat()
        #   encoder4 + decoder4 = decoder4
        self.conv_concat4 = ConvModule(in_channels=512,
                                       out_channels=embed_dims[3], kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_concat4 = Linear(input_dim=embed_dims[3], embed_dim=embed_dims[3])

        #   encoder4 + decoder4 = decoder3
        self.conv_concat34 = ConvModule(in_channels=(256 + embed_dims[3]),
                                        out_channels=embed_dims[2], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_concat34 = Linear(input_dim=embed_dims[2], embed_dim=embed_dims[2])

        #   encoder3 + decoder3 = decoder2
        self.conv_concat23 = ConvModule(in_channels=(embed_dims[1] + embed_dims[2]),
                                        out_channels=embed_dims[1], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_concat23 = Linear(input_dim=embed_dims[1], embed_dim=embed_dims[1])

        #   encoder2 + decoder2 = decoder1
        self.conv_concat12 = ConvModule(in_channels=(embed_dims[0] + embed_dims[1]),
                                        out_channels=embed_dims[0], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_concat12 = Linear(input_dim=embed_dims[0], embed_dim=embed_dims[0])

    def forward(self, encoder):  # encoder = encoder1, encoder2, cl_encoder3, cl_encoder4
        encoder1, encoder2, encoder3, encoder4 = encoder
        #   decoder4
        decoder4 = encoder4  # 16, 256, 22, 22
        decoder4 = self.skip_concat(encoder4, decoder4)  # 16, 512, 22, 22
        decoder4 = self.linear_concat4(self.conv_concat4(decoder4))  # 16, 512, 22, 22

        #   decoder3
        decoder3 = self.skip_concat(encoder3, decoder4)  # 16, 768, 22, 22
        decoder3 = self.linear_concat34(self.conv_concat34(decoder3))  # 16, 320, 22, 22

        #   decoder2
        decoder3_up2 = F.interpolate(decoder3, scale_factor=2, mode='bilinear', align_corners=False)
        decoder2 = self.skip_concat(encoder2, decoder3_up2)  # 16, 448, 44, 44
        decoder2 = self.linear_concat23(self.conv_concat23(decoder2))  # 16, 128, 44, 44

        #   decoder1
        decoder2_up2 = F.interpolate(decoder2, scale_factor=2, mode='bilinear', align_corners=False)
        decoder1 = self.skip_concat(encoder1, decoder2_up2)  # 16, 192, 88, 88
        decoder1 = self.linear_concat12(self.conv_concat12(decoder1))  # 16, 64, 88, 88

        decoder = [decoder1, decoder2, decoder3, decoder4]
        return decoder
