import torch.nn as nn
import torch.nn.functional as F

from .cl import Contrastive_loss
from .utils import Linear


class Encoder_1(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512]):
        super(Encoder, self).__init__()
        self.linear_encoder1 = Linear(input_dim=embed_dims[0], embed_dim=embed_dims[0])
        self.linear_encoder2 = Linear(input_dim=embed_dims[1], embed_dim=embed_dims[1])
        self.linear_encoder3 = Linear(input_dim=embed_dims[2], embed_dim=embed_dims[2])
        self.linear_encoder4 = Linear(input_dim=embed_dims[3], embed_dim=embed_dims[3])

    def forward(self, backbone):
        encoder = [encoder1, encoder2, encoder3, encoder4]

        return encoder

class Encoder_2(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512]):
        super(Encoder, self).__init__()
        self.linear_encoder1 = Linear(input_dim=embed_dims[0], embed_dim=embed_dims[0])
        self.linear_encoder2 = Linear(input_dim=embed_dims[1], embed_dim=embed_dims[1])
        self.linear_encoder3 = Linear(input_dim=embed_dims[2], embed_dim=embed_dims[2])
        self.linear_encoder4 = Linear(input_dim=embed_dims[3], embed_dim=embed_dims[3])

    def forward(self, backbone):
        encoder1 = self.linear_encoder1(backbone[0])  # 16, 64, 88, 88
        encoder2 = self.linear_encoder2(backbone[1])  # 16, 128, 44, 44
        encoder3 = self.linear_encoder3(backbone[2])  # 16, 320, 22, 22
        encoder4 = self.linear_encoder4(backbone[3])  # 16, 512, 11, 11
        encoder = [encoder1, encoder2, encoder3, encoder4]

        return encoder


class Cl_Encoder(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512]):
        super(Cl_Encoder, self).__init__()
        self.cl = Contrastive_loss(0.5)
        self.Linear3 = Linear(input_dim=embed_dims[2], embed_dim=256)
        self.Linear4 = Linear(input_dim=embed_dims[3], embed_dim=256)
        self.cl_linear1 = Linear(256, 64)
        self.cl_linear2 = Linear(64, 256)

    def forward(self, encoder):  # encoder = [encoder1, encoder2, encoder3, encoder4]
        encoder1, encoder2, encoder3, encoder4 = encoder
        """ representation"""
        #   encoder3
        cl_encoder3 = self.Linear3(encoder3)
        cl_encoder3 = F.relu(self.cl_linear1(cl_encoder3))
        cl_encoder3 = self.cl_linear2(cl_encoder3)
        #   encoder4
        cl_encoder4 = F.interpolate(encoder4, scale_factor=2, mode='bilinear', align_corners=False)
        cl_encoder4 = self.Linear4(cl_encoder4)
        cl_encoder4 = F.relu(self.cl_linear1(cl_encoder4))
        cl_encoder4 = self.cl_linear2(cl_encoder4)

        """ loss """
        cl_encoder34 = self.cl(cl_encoder3, cl_encoder4)

        """ return """
        encoder = encoder1, encoder2, cl_encoder3, cl_encoder4
        contrast_loss = cl_encoder34

        return encoder, contrast_loss
