import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

class Skip_Concat(nn.Module):
    def __init__(self):
        super(Skip_Concat, self).__init__()

    def forward(self, encoder, decoder):
        res = torch.cat([encoder, decoder], dim=1)
        return res
