import torch
import torch.nn as nn


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions.
    From https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/utils.py"""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def norm_thresholding(x0, value):
    s = append_dims(x0.pow(2).flatten(1).mean(1).sqrt().clamp(min=value), x0.ndim)
    return x0 * (value / s)


def spatial_norm_thresholding(x0, value):
    # b c h w
    s = x0.pow(2).mean(1, keepdim=True).sqrt().clamp(min=value)
    return x0 * (value / s)


class Gauss_Shifts(nn.Module):
    def __init__(self, shape, save_ckpt_path=None,
                 ckpt_path=None, mean_path=None, std_path=None):
        super(Gauss_Shifts, self).__init__()
        self.cnt = 1
        self.shape = tuple(shape)
        self.sample_gauss_std = nn.Parameter(torch.ones(self.shape), requires_grad=False)
        self.sample_gauss_mean = nn.Parameter(torch.zeros(self.shape), requires_grad=False)
        self.keep_gauss_mean = self.sample_gauss_mean.detach().clone()
        self.save_ckpt_path = save_ckpt_path
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            state_dict = state['state_dict']
            self.load_state_dict(state_dict)

    def save_ckpt(self):
        self.push()
        state = self.state_dict()
        torch.save({'state_dict': state}, self.save_ckpt_path)

    @torch.no_grad()
    def forward(self, x):
        # x: [b *shape]
        n = x.shape[0]
        x_mean = torch.sum(x, dim=0) / n
        cur_mean = self.keep_gauss_mean.data
        x_std = (x - cur_mean.unsqueeze(0))**2
        x_std = torch.sum(x_std, dim=0) / n # c h w
        self.cnt += 1
        self.sample_gauss_std.data += x_std
        self.sample_gauss_mean.data += x_mean

    @torch.no_grad()
    def push(self):
        n = self.cnt
        self.cnt = 1
        self.sample_gauss_std /= n
        self.sample_gauss_mean /= n
        self.keep_gauss_mean = self.sample_gauss_mean.detach().clone()

    @torch.no_grad()
    def pull_mean(self, device=None):
        return self.sample_gauss_mean.detach().data

    @torch.no_grad()
    def pull_std(self, device=None):
        return torch.sqrt(self.sample_gauss_std.detach().data)

