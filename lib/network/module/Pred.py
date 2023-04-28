import torch
import torch.nn as nn
import torch.nn.functional as F


class Out_Pred(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512]):
        super(Out_Pred, self).__init__()
        self.conv_pred = nn.Conv2d((embed_dims[0] + embed_dims[1] +
                                    embed_dims[2]), 1, kernel_size=1)

    def forward(self, decoder):  # decoder = [decoder1, decoder2, decoder3, decoder4]
        decoder1, decoder2, decoder3, decoder4 = decoder

        decoder1 = decoder1  # 16, 64, 88, 88
        decoder2 = F.interpolate(decoder2, scale_factor=2, mode='bilinear', align_corners=False)  # 16, 128, 88, 88
        decoder3 = F.interpolate(decoder3, scale_factor=4, mode='bilinear', align_corners=False)  # 16, 320, 88, 88

        pred = self.conv_pred(torch.cat([decoder3, decoder2, decoder1], dim=1))
        return pred
