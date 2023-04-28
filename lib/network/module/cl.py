#   Gross
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
        z = F.elu(self.fc1(z))
        return self.fc2(z)

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


class CL_spatial_local(nn.Module):
    def __init__(self, tau=0.4, num_channel=256, num_proj_hidden=64):
        super().__init__()
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_channel, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_channel)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = rearrange(z, 'b c h w -> b (h w) c')
        z = F.elu(self.fc1(z))
        z = self.fc2(z)
        return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=2)
        return torch.matmul(z1, z2.transpose(1, 2))

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        exp = lambda x: torch.exp(x / self.tau)

        intra_sim = exp(self.sim(z1, z1))
        intra_sim_sum = torch.sum(intra_sim, dim=2)  # AA + AB
        intra_sim_diag = torch.diagonal(intra_sim, dim1=1, dim2=2)  # AA

        inter_sim = exp(self.sim(z1, z2))
        inter_sim_sum = torch.sum(inter_sim, dim=2)  # Aa + Ab
        inter_sim_diag = torch.diagonal(inter_sim, dim1=1, dim2=2)  # Aa

        #   对应spatial / 对应spatial + 同图片其他spatial + 对面图片其他spatial
        numerator = inter_sim_diag  # Aa
        denominator = intra_sim_sum + inter_sim_sum - intra_sim_diag  # AA + AB + Aa + Ab - AA
        res = -torch.log(numerator / denominator)  # Aa / (Aa + AB + Ab)
        return res

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret


class CL_channel_local(nn.Module):
    def __init__(self, tau=0.4, num_channel=256, num_proj_hidden=64):
        super().__init__()
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_channel, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_channel)

    #   channel和spatial就差这了，但是应该不是这么简单，channel应该是整张图片的特征分布才对，留着以后方便改所以没和spatial统一
    def projection(self, z: torch.Tensor):
        z = rearrange(z, 'b c h w -> b (h w) c')
        z = F.elu(self.fc1(z))
        z = self.fc2(z)
        return z.transpose(1, 2)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=2)
        return torch.matmul(z1, z2.transpose(1, 2))

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        exp = lambda x: torch.exp(x / self.tau)

        intra_sim = exp(self.sim(z1, z1))
        intra_sim_sum = torch.sum(intra_sim, dim=2)  # AA + AB
        intra_sim_diag = torch.diagonal(intra_sim, dim1=1, dim2=2)  # AA

        inter_sim = exp(self.sim(z1, z2))
        inter_sim_sum = torch.sum(inter_sim, dim=2)  # Aa + Ab
        inter_sim_diag = torch.diagonal(inter_sim, dim1=1, dim2=2)  # Aa

        #   对应channel / 对应channel + 同图片其他channel + 对面图片其他channel
        numerator = inter_sim_diag  # Aa
        denominator = intra_sim_sum + inter_sim_sum - intra_sim_diag  # AA + AB + Aa + Ab - AA
        res = -torch.log(numerator / denominator)  # Aa / (Aa + AB + Ab)
        return res

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret


class CL_spatial_global(nn.Module):
    def __init__(self, tau=0.4, num_channel=256, num_proj_hidden=64):
        super().__init__()
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_channel, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_channel)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = rearrange(z, 'b c h w -> b (h w) c')
        z = F.elu(self.fc1(z))
        z = self.fc2(z)
        return z

    def sim_global(self, z1: torch.Tensor, z2: torch.Tensor):  # 在外面进行的转置所以对列取模
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=0)
        return torch.matmul(z1, z2)

    def sim_local(self, z1: torch.Tensor, z2: torch.Tensor):  # 在外面没有转置
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=2)
        return torch.matmul(z1, z2.transpose(1, 2))

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        B, N, C = z1.shape
        exp = lambda x: torch.exp(x / self.tau)

        #   local
        intra_sim_local = exp(self.sim_local(z1, z1))
        intra_sim_local_diag = torch.diagonal(intra_sim_local, dim1=1, dim2=2)  # AA
        inter_sim_local = exp(self.sim_local(z1, z2))
        inter_sim_local_diag = torch.diagonal(inter_sim_local, dim1=1, dim2=2)  # Aa

        #   global
        if B > 1:
            intra_z1 = torch.cat([z1, z1], dim=2)  # z1 * B
            for i in range(B - 2):
                intra_z1 = torch.cat([intra_z1, z1], dim=2)  # z1 * B
        else:
            intra_z1 = z1
        inter_z1 = z1.transpose(1, 2).reshape(B * C, N)
        inter_z2 = z2.transpose(1, 2).reshape(B * C, N)

        intra_sim_global = exp(self.sim_global(intra_z1, inter_z1))
        intra_sim_global_sum = torch.sum(intra_sim_global, dim=2)  # Aa + Ac + Ab + Ad

        inter_sim_global = exp(self.sim_global(intra_z1, inter_z2))
        inter_sim_global_sum = torch.sum(inter_sim_global, dim=2)  # AA + AC + AB + AD

        numerator = inter_sim_local_diag  # Aa
        denominator = intra_sim_global_sum + inter_sim_global_sum - intra_sim_local_diag  # Aa + Ab + Ac + Ad + AA + AB + AC + AD - AA
        loss = -torch.log(numerator / denominator)
        return loss

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret


class CL_channel_global(nn.Module):
    def __init__(self, tau=0.4, num_channel=256, num_proj_hidden=64):
        super().__init__()
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_channel, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_channel)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = rearrange(z, 'b c h w -> b (h w) c')
        z = F.elu(self.fc1(z))
        z = self.fc2(z)
        return z.transpose(1, 2)

    def sim_global(self, z1: torch.Tensor, z2: torch.Tensor):  # 在外面进行的转置所以对列取模
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=0)
        return torch.matmul(z1, z2)

    def sim_local(self, z1: torch.Tensor, z2: torch.Tensor):  # 在外面没有转置
        z1 = F.normalize(z1, dim=2)
        z2 = F.normalize(z2, dim=2)
        return torch.matmul(z1, z2.transpose(1, 2))

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        B, N, C = z1.shape
        exp = lambda x: torch.exp(x / self.tau)

        #   local
        intra_sim_local = exp(self.sim_local(z1, z1))
        intra_sim_local_diag = torch.diagonal(intra_sim_local, dim1=1, dim2=2)  # AA
        inter_sim_local = exp(self.sim_local(z1, z2))
        inter_sim_local_diag = torch.diagonal(inter_sim_local, dim1=1, dim2=2)  # Aa

        #   global
        if B > 1:
            intra_z1 = torch.cat([z1, z1], dim=2)  # z1 * B
            for i in range(B - 2):
                intra_z1 = torch.cat([intra_z1, z1], dim=2)  # z1 * B
        else:
            intra_z1 = z1
        inter_z1 = z1.transpose(1, 2).reshape(B * C, N)
        inter_z2 = z2.transpose(1, 2).reshape(B * C, N)

        intra_sim_global = exp(self.sim_global(intra_z1, inter_z1))
        intra_sim_global_sum = torch.sum(intra_sim_global, dim=2)  # Aa + Ac + Ab + Ad

        inter_sim_global = exp(self.sim_global(intra_z1, inter_z2))
        inter_sim_global_sum = torch.sum(inter_sim_global, dim=2)  # AA + AC + AB + AD

        numerator = inter_sim_local_diag  # Aa
        denominator = intra_sim_global_sum + inter_sim_global_sum - intra_sim_local_diag  # Aa + Ab + Ac + Ad + AA + AB + AC + AD - AA
        loss = -torch.log(numerator / denominator)
        return loss

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret


if __name__ == '__main__':
    cl_s_l = CL_spatial_local()
    cl_c_l = CL_channel_local()
    cl_s_g = CL_spatial_global()
    cl_c_g = CL_channel_global()
    z1 = torch.randn(16, 256, 44, 44)
    z2 = torch.randn(16, 256, 44, 44)
    out_s_l = cl_s_l(z1, z2)
    out_c_l = cl_c_l(z1, z2)
    out_s_g = cl_s_g(z1, z2)
    out_c_g = cl_c_g(z1, z2)

    print(out)
