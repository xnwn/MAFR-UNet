import torch
import torch.nn as nn

from .layers import Mlp
from .utils import LePEAttention
from timm.layers import DropPath


class CSWinBlock(nn.Module):
    def __init__(self, patch_resolution, normalization, qkv_bias, stripe_width, num_heads, dim, qk_scale,
                 qk_dropout_probability, drop_path_probability, mlp_ratio, dropout_probability, mlp_activation=nn.GELU,
                 last_encoder=False):
        super().__init__()
        self.patch_resolution = patch_resolution
        self.normalization_1 = normalization(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        if last_encoder:
            self.branch_num = 1
            self.attention = nn.ModuleList([
                LePEAttention(patch_resolution, -1, stripe_width, num_heads, dim, qk_scale, qk_dropout_probability)
                for _ in range(self.branch_num)])
        else:
            self.branch_num = 2
            self.attention = nn.ModuleList([
                LePEAttention(patch_resolution, i, stripe_width, num_heads // 2, dim // 2, qk_scale,
                              qk_dropout_probability)
                for i in range(self.branch_num)])

        self.linear = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path_probability) if drop_path_probability > 0. else nn.Identity()
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, mlp_activation, dropout_probability)
        self.normalization_2 = normalization(dim)

    def forward(self, x):
        height = width = self.patch_resolution
        batch, length, channel = x.shape
        assert length == height * width, "flatten img_tokens has wrong size"

        img = self.normalization_1(x)
        # batch*length*channel -> batch*length*3*channel -> 3*batch*length*channel
        qkv = self.qkv(img).reshape(batch, -1, 3, channel).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attention[0](qkv[:, :, :, :channel // 2])
            x2 = self.attention[1](qkv[:, :, :, channel // 2:])
            attention_out = torch.cat([x1, x2], dim=2)
        else:
            attention_out = self.attention[0](qkv)

        x = x + self.drop_path(self.linear(attention_out))
        x = x + self.drop_path(self.mlp(self.normalization_2(x)))

        return x


class BottleneckConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, middle_channels, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.p_relu1 = nn.PReLU(num_parameters=middle_channels)

        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.p_relu2 = nn.PReLU(num_parameters=middle_channels)

        self.conv3 = nn.Conv2d(middle_channels, out_channels, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.p_relu3 = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.p_relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.p_relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.p_relu3(out)

        return out


class SkipConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.p_relu1 = nn.PReLU(num_parameters=middle_channels)

        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.p_relu2 = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.p_relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.p_relu2(out)

        return out
