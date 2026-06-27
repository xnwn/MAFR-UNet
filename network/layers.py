import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import nlp2conv


class Mlp(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mlp_activate, dropout_probability):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_channels)
        self.activate = mlp_activate()
        self.linear_2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activate(x)
        x = self.dropout(x)

        x = self.linear_2(x)
        x = self.dropout(x)

        return x


class ConvMerge(nn.Module):
    def __init__(self, in_channels, out_channels, normalization=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.normalization = normalization(out_channels)

    def forward(self, x):
        batch, height, width, channel = nlp2conv(x)
        x = x.transpose(-2, -1).contiguous().view(batch, channel, height, width)
        x = self.conv(x)

        batch, channel = x.shape[:2]
        x = x.view(batch, channel, -1).transpose(-2, -1).contiguous()

        x = self.normalization(x)

        return x


class CARAFE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, up_factor=2, conv=False):
        super().__init__()
        self.conv = conv
        self.channel_down = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.encoder = nn.Conv2d(in_channels // 4, up_factor ** 2 * kernel_size ** 2, kernel_size, 1, kernel_size // 2)
        self.up_factor = up_factor
        self.kernel_size = kernel_size
        self.out = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        if not self.conv:
            batch, height, width, channel = nlp2conv(x)
            x = x.transpose(-2, -1).contiguous().view(batch, channel, height, width)
        else:
            batch, channel, height, width = x.shape

        kernel_tensor = self.channel_down(x)
        kernel_tensor = self.encoder(kernel_tensor)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.reshape(batch, self.kernel_size ** 2, height, width, self.up_factor ** 2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)

        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)
        w = w.unfold(2, self.kernel_size, step=1)
        w = w.unfold(3, self.kernel_size, step=1)
        w = w.reshape(batch, channel, height, width, -1)
        w = w.permute(0, 2, 3, 1, 4)

        x = torch.matmul(w, kernel_tensor)
        x = x.reshape(batch, height, width, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)

        x = self.out(x)

        if not self.conv:
            batch, channel = x.shape[:2]
            x = x.view(batch, channel, -1).transpose(-2, -1).contiguous()

        return x
