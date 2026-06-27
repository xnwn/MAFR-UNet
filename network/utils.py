import numpy as np
import torch.nn as nn


def img2windows(img, height_sw, width_sw):
    batch, channel, height, width = img.shape
    img_reshape = img.view(batch, channel, height // height_sw, height_sw, width // width_sw, width_sw)
    img_window = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, height_sw * width_sw, channel)
    return img_window


def windows2img(img, height_sw, width_sw, height, width):
    batch = int(img.shape[0] / (height * width / height_sw / width_sw))
    img = img.view(batch, height // height_sw, width // width_sw, height_sw, width_sw, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)
    return img


class LePEAttention(nn.Module):
    def __init__(self, patch_resolution, idx, stripe_width, num_heads, dim, qk_scale, qk_dropout_probability):
        super().__init__()
        self.patch_resolution = patch_resolution

        if idx == -1:
            self.height_sw, self.width_sw = patch_resolution, patch_resolution
        elif idx == 0:
            self.height_sw, self.width_sw = patch_resolution, stripe_width
        elif idx == 1:
            self.width_sw, self.height_sw = patch_resolution, stripe_width
        else:
            print("ERROR MODE", idx)
            exit(0)

        self.num_heads = num_heads
        self.calculate_pe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qk_dropout = nn.Dropout(qk_dropout_probability)

    def img2cswin(self, x):
        batch, length, channel = x.shape
        height = width = int(np.sqrt(length))

        x = x.transpose(-2, -1).contiguous().view(batch, channel, height, width)
        x = img2windows(x, self.height_sw, self.width_sw)
        x = x.reshape(-1, self.height_sw * self.width_sw,
                      self.num_heads, channel // self.num_heads).permute(0, 2, 1, 3).contiguous()

        return x

    def get_local_enhance_pe(self, x, calculate_pe):
        batch, length, channel = x.shape
        height = width = int(np.sqrt(length))

        x = x.transpose(-2, -1).contiguous().view(batch, channel, height, width)
        height_sw, width_sw = self.height_sw, self.width_sw
        x = x.view(batch, channel, height // height_sw, height_sw, width // width_sw, width_sw)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, channel, height_sw, width_sw)

        local_enhance_pe = calculate_pe(x)
        local_enhance_pe = local_enhance_pe.reshape(-1, self.num_heads, channel // self.num_heads,
                                                    height_sw * width_sw).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, channel // self.num_heads,
                      self.height_sw * self.width_sw).permute(0, 1, 3, 2).contiguous()

        return x, local_enhance_pe

    def forward(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]
        height = width = self.patch_resolution
        batch, length, channel = q.shape
        assert length == height * width, "flatten img_tokens has wrong size"

        q = self.img2cswin(q)
        k = self.img2cswin(k)
        v, local_enhance_pe = self.get_local_enhance_pe(v, self.calculate_pe)
        q = q * self.scale

        qk = (q @ k.transpose(-2, -1))
        qk = nn.functional.softmax(qk, dim=-1, dtype=qk.dtype)
        qk = self.qk_dropout(qk)

        x = (qk @ v) + local_enhance_pe
        x = x.transpose(1, 2).reshape(-1, self.height_sw * self.width_sw, channel)
        x = windows2img(x, self.height_sw, self.width_sw, height, width).view(batch, -1, channel)

        return x


def nlp2conv(x):
    batch, length, channel = x.shape
    height = width = int(np.sqrt(length))
    return batch, height, width, channel
