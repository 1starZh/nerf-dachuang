import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Sinusoidal Positional Encoding（原始 NeRF 位置编码）
def sinusoidal_encoding(x, L=10):
    """
    Args:
        x: [..., D] coordinates (e.g., 3D)
        L: number of frequency bands
    Returns:
        encoded: [..., D * 2 * L]
    """
    freq_bands = 2. ** torch.linspace(0., L - 1, L)
    x_expanded = (x[..., None, :] * freq_bands[:, None])  # [..., L, D]
    encoded = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)  # [..., L, 2D]
    return encoded.view(*x.shape[:-1], -1)

# 几何增强编码（结合先验）
def enhanced_encoding(pos, prior=None, L=10):
    """
    结合位置编码与几何先验，例如法向量或平面法线等。

    Args:
        pos: [..., 3] 位置坐标（x, y, z）
        prior: [..., D] 几何先验（如法向量、局部平面等）
        L: 编码频率层数

    Returns:
        [..., N] 组合后的编码
    """
    pos_enc = sinusoidal_encoding(pos, L)
    if prior is not None:
        prior = F.normalize(prior, dim=-1)  # 保持单位长度
        return torch.cat([pos_enc, prior], dim=-1)
    else:
        return pos_enc

# 用于替代原始 NeRF 的 MLP 网络，支持输入几何先验
class NeRFWithPriors(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_prior=3, input_ch_dir=27, output_ch=4, skips=[4]):
        super(NeRFWithPriors, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_prior = input_ch_prior
        self.input_ch_dir = input_ch_dir
        self.skips = skips

        in_channels = input_ch + input_ch_prior

        self.pts_linears = nn.ModuleList(
            [nn.Linear(in_channels, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + in_channels, W) for i in range(D - 1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_dir + W, W // 2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x):
        """
        Args:
            x: [..., input_ch + input_ch_prior + input_ch_dir]
        Returns:
            raw: [..., 4] (r, g, b, sigma)
        """
        input_pts, input_dirs = torch.split(x, [self.input_ch + self.input_ch_prior, self.input_ch_dir], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        h = torch.cat([feature, input_dirs], -1)

        for l in self.views_linears:
            h = l(h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs
    
    