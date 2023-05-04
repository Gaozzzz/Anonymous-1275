import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Contrastive_loss(nn.Module):
    def __init__(self, tau=0.4, num_channel=256, num_proj_hidden=64):
        super(Contrastive_loss, self).__init__()
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_channel, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_channel)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = rearrange(z, 'b c h w -> b (h w) c')
        return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=2)
        return torch.matmul(z1, z2.transpose(1, 2))

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        refl_sim_sum = torch.sum(refl_sim, dim=2)
        refl_sim_diag = torch.diagonal(refl_sim, dim1=1, dim2=2)
        between_sim_diag = torch.diagonal(between_sim, dim1=1, dim2=2)
        between_sim_sum = torch.sum(between_sim, dim=2)

        numerator = between_sim_diag
        denominator = refl_sim_sum + between_sim_sum - refl_sim_diag
        res = -torch.log(numerator / denominator)
        return res

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret