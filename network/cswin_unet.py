import numpy as np
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from network.blocks import BottleneckConvBlock, CSWinBlock, SkipConvBlock
from network.layers import CARAFE, ConvMerge
from network.utils import nlp2conv
from timm.layers import trunc_normal_


class CSWinUnet(nn.Module):
    def __init__(self, in_channels, embed_dim, img_size, dropout_probability, block_num_list, drop_path_probability,
                 qkv_bias, stripe_width_list, num_heads_list, qk_scale, mlp_ratio, num_classes,
                 qk_dropout_probability=0., normalization=nn.LayerNorm):
        super().__init__()

        self.conv_token_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim)
        )
        self.token_dropout = nn.Dropout(p=dropout_probability)

        current_dim = embed_dim
        block_num_count = int(np.sum(block_num_list))
        drop_path_probability_list = [x.item() for x in torch.linspace(0, drop_path_probability, block_num_count)]

        self.encoder_1 = nn.ModuleList([CSWinBlock(img_size // 4, normalization, qkv_bias, stripe_width_list[0],
                                                   num_heads_list[0], current_dim, qk_scale, qk_dropout_probability,
                                                   drop_path_probability_list[i], mlp_ratio, dropout_probability)
                                        for i in range(block_num_list[0])])
        self.conv_merge_1 = ConvMerge(current_dim, current_dim * 2)
        current_dim = current_dim * 2

        self.encoder_2 = nn.ModuleList([CSWinBlock(img_size // 8, normalization, qkv_bias, stripe_width_list[1],
                                                   num_heads_list[1], current_dim, qk_scale, qk_dropout_probability,
                                                   drop_path_probability_list[np.sum(block_num_list[:1]) + i],
                                                   mlp_ratio, dropout_probability)
                                        for i in range(block_num_list[1])])
        self.conv_merge_2 = ConvMerge(current_dim, current_dim * 2)
        current_dim = current_dim * 2

        self.encoder_3 = nn.ModuleList([CSWinBlock(img_size // 16, normalization, qkv_bias, stripe_width_list[2],
                                                   num_heads_list[2], current_dim, qk_scale, qk_dropout_probability,
                                                   drop_path_probability_list[np.sum(block_num_list[:2]) + i],
                                                   mlp_ratio, dropout_probability)
                                        for i in range(block_num_list[2])])
        self.conv_merge_3 = ConvMerge(current_dim, current_dim * 2)
        current_dim = current_dim * 2

        self.encoder_norm = normalization(current_dim)

        self.bottleneckConvBlock = BottleneckConvBlock(512, 128, 512)

        channel_list = [64, 128, 256, 512]

        self.conv0_1 = SkipConvBlock(channel_list[0] * 2, channel_list[0], channel_list[0])
        self.conv1_1 = SkipConvBlock(channel_list[1] * 2, channel_list[1], channel_list[1])
        self.conv2_1 = SkipConvBlock(channel_list[2] * 2, channel_list[2], channel_list[2])
        self.up_sample1 = CARAFE(channel_list[1], channel_list[0], conv=True)

        self.conv0_2 = SkipConvBlock(channel_list[0] * 3, channel_list[0], channel_list[0])
        self.conv1_2 = SkipConvBlock(channel_list[1] * 3, channel_list[1], channel_list[1])
        self.up_sample2 = CARAFE(channel_list[2], channel_list[1], conv=True)

        self.conv0_3 = SkipConvBlock(channel_list[0] * 4, channel_list[0], channel_list[0])
        self.up_sample3 = CARAFE(channel_list[3], channel_list[2], conv=True)

        self.carafe_3 = CARAFE(current_dim, current_dim // 2)
        self.concat_linear_3 = nn.Linear(current_dim, current_dim // 2)
        current_dim = current_dim // 2
        self.decoder_3 = nn.ModuleList([CSWinBlock(img_size // 16, normalization, qkv_bias, stripe_width_list[2],
                                                   num_heads_list[2], current_dim, qk_scale, qk_dropout_probability,
                                                   drop_path_probability_list[np.sum(block_num_list[:2]) + i],
                                                   mlp_ratio, dropout_probability)
                                        for i in range(block_num_list[2])])

        self.carafe_2 = CARAFE(current_dim, current_dim // 2)
        self.concat_linear_2 = nn.Linear(current_dim, current_dim // 2)
        current_dim = current_dim // 2
        self.decoder_2 = nn.ModuleList([CSWinBlock(img_size // 8, normalization, qkv_bias, stripe_width_list[1],
                                                   num_heads_list[1], current_dim, qk_scale, qk_dropout_probability,
                                                   drop_path_probability_list[np.sum(block_num_list[:1]) + i],
                                                   mlp_ratio, dropout_probability)
                                        for i in range(block_num_list[1])])

        self.carafe_1 = CARAFE(current_dim, current_dim // 2)
        self.concat_linear_1 = nn.Linear(current_dim, current_dim // 2)
        current_dim = current_dim // 2
        self.decoder_1 = nn.ModuleList([CSWinBlock(img_size // 4, normalization, qkv_bias, stripe_width_list[0],
                                                   num_heads_list[0], current_dim, qk_scale, qk_dropout_probability,
                                                   drop_path_probability_list[i], mlp_ratio, dropout_probability)
                                        for i in range(block_num_list[0])])

        self.decoder_norm = normalization(embed_dim)

        self.carafe_up = CARAFE(current_dim, embed_dim, up_factor=4)

        self.channel_covert = nn.Conv2d(embed_dim, num_classes, 1, bias=False)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def encoder(self, x):
        x = self.conv_token_embedding(x)
        x = self.token_dropout(x)

        for block in self.encoder_1:
            x = block(x)
        batch, height, width, channel = nlp2conv(x)
        skip_connection_temp = [x.view(batch, height, width, channel).permute(0, 3, 1, 2)]
        x = self.conv_merge_1(x)

        for block in self.encoder_2:
            x = block(x)
        batch, height, width, channel = nlp2conv(x)
        skip_connection_temp.append(x.view(batch, height, width, channel).permute(0, 3, 1, 2))
        x = self.conv_merge_2(x)

        for block in self.encoder_3:
            x = block(x)
        batch, height, width, channel = nlp2conv(x)
        skip_connection_temp.append(x.view(batch, height, width, channel).permute(0, 3, 1, 2))
        x = self.conv_merge_3(x)

        x = self.encoder_norm(x)

        return x, skip_connection_temp

    def bottleneck(self, x, skip_connection_temp):
        batch, height, width, channel = nlp2conv(x)
        x = x.reshape(batch, height, width, channel).permute(0, 3, 1, 2)

        x = self.bottleneckConvBlock(x)

        skip_connection_temp.append(x)

        x = x.permute(0, 2, 3, 1)
        batch, height, width, channel = x.shape
        x = x.reshape(batch, height * width, channel)

        return x, skip_connection_temp

    def skip_connection(self, skip_connection_temp):
        x0_1 = self.conv0_1(torch.cat([skip_connection_temp[0], self.up_sample1(skip_connection_temp[1])], 1))
        x1_1 = self.conv1_1(torch.cat([skip_connection_temp[1], self.up_sample2(skip_connection_temp[2])], 1))
        x2_1 = self.conv2_1(torch.cat([skip_connection_temp[2], self.up_sample3(skip_connection_temp[3])], 1))

        x0_2 = self.conv0_2(torch.cat([skip_connection_temp[0], x0_1, self.up_sample1(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([skip_connection_temp[1], x1_1, self.up_sample2(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([skip_connection_temp[0], x0_1, x0_2, self.up_sample1(x1_2)], 1))

        return [torch.flatten(x0_3, start_dim=2, end_dim=-1).permute(0, 2, 1),
                torch.flatten(x1_2, start_dim=2, end_dim=-1).permute(0, 2, 1),
                torch.flatten(x2_1, start_dim=2, end_dim=-1).permute(0, 2, 1)]

    def decoder(self, x, skip_connection_temp):
        x = self.carafe_3(x)
        x = torch.cat([skip_connection_temp[-1], x], -1)
        x = self.concat_linear_3(x)
        for block in self.decoder_3:
            x = block(x)

        x = self.carafe_2(x)
        x = torch.cat([skip_connection_temp[-2], x], -1)
        x = self.concat_linear_2(x)
        for block in self.decoder_2:
            x = block(x)

        x = self.carafe_1(x)
        x = torch.cat([skip_connection_temp[-3], x], -1)
        x = self.concat_linear_1(x)
        for block in self.decoder_1:
            x = block(x)

        x = self.decoder_norm(x)

        batch, length, channel = x.shape
        height = weight = int(np.sqrt(length))
        x = self.carafe_up(x)
        x = x.view(batch, 4 * height, 4 * weight, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.channel_covert(x)

        return x

    def forward(self, x):
        x, skip_connection_temp = self.encoder(x)
        x, skip_connection_temp = self.bottleneck(x, skip_connection_temp)
        skip_connection_temp = self.skip_connection(skip_connection_temp)
        x = self.decoder(x, skip_connection_temp)

        return x
