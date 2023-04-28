import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Linear Prediction 用mlp对channel进行操作
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


class BasicConv2d(nn.Module):  # CNN+BN，but no res
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)  # ？

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class show(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_backbone1 = BasicConv2d(64, 1, kernel_size=1, padding=1)
        self.out_backbone2 = BasicConv2d(128, 1, kernel_size=1, padding=1)
        self.out_backbone3 = BasicConv2d(320, 1, kernel_size=1, padding=1)
        self.out_backbone4 = BasicConv2d(512, 1, kernel_size=1, padding=1)

        self.out_encoder1 = BasicConv2d(64, 1, kernel_size=1, padding=1)
        self.out_encoder2 = BasicConv2d(128, 1, kernel_size=1, padding=1)
        self.out_encoder3 = BasicConv2d(256, 1, kernel_size=1, padding=1)
        self.out_encoder4 = BasicConv2d(256, 1, kernel_size=1, padding=1)

        self.out_decoder1 = BasicConv2d(64, 1, kernel_size=1, padding=1)
        self.out_decoder2 = BasicConv2d(128, 1, kernel_size=1, padding=1)
        self.out_decoder3 = BasicConv2d(320, 1, kernel_size=1, padding=1)
        self.out_decoder4 = BasicConv2d(256, 1, kernel_size=1, padding=1)

    def forward(self, backbone, encoder, decoder):
        backbone1, backbone2, backbone3, backbone4 = backbone
        encoder1, encoder2, encoder3, encoder4 = encoder
        decoder1, decoder2, decoder3, decoder4 = decoder
        out_backbone1 = self.out_backbone1(backbone1)
        out_backbone2 = self.out_backbone2(backbone2)
        out_backbone3 = self.out_backbone3(backbone3)
        out_backbone4 = self.out_backbone4(backbone4)

        out_encoder1 = self.out_encoder1(encoder1)
        out_encoder2 = self.out_encoder2(encoder2)
        out_encoder3 = self.out_encoder3(encoder3)
        out_encoder4 = self.out_encoder4(encoder4)

        out_decoder1 = self.out_decoder1(decoder1)
        out_decoder2 = self.out_decoder2(decoder2)
        out_decoder3 = self.out_decoder3(decoder3)
        out_decoder4 = self.out_decoder4(decoder4)

        debug_backbone = [out_backbone1, out_encoder1, out_backbone2, out_encoder2, out_backbone3, out_encoder3,
                          out_backbone4, out_encoder4]
        debug_encoder = [out_encoder1, out_decoder1, out_encoder2, out_decoder2, out_encoder3, out_decoder3,
                         out_encoder4, out_decoder4]
        debug_docoder = [out_decoder1, out_decoder2, out_decoder3, out_decoder4]
        debug_concat = [out_decoder4, out_encoder3, out_decoder3, out_encoder2, out_decoder2, out_encoder1,
                        out_decoder1]

        return debug_backbone, debug_encoder, debug_docoder, debug_concat


class Skip_Concat(nn.Module):
    def __init__(self):
        super(Skip_Concat, self).__init__()

    def forward(self, encoder, decoder):
        res = torch.cat([encoder, decoder], dim=1)
        return res
