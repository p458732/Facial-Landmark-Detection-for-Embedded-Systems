"""
EfficientFormer_v2
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict
import itertools

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple


from torchstat import stat

class E2HTransform(nn.Module):
    def __init__(self, edge_info, num_points, num_edges):
        super().__init__()

        e2h_matrix = np.zeros([num_points, num_edges])
        for edge_id, isclosed_indices in enumerate(edge_info):
            is_closed, indices = isclosed_indices
            for point_id in indices:
                e2h_matrix[point_id, edge_id] = 1
        e2h_matrix = torch.from_numpy(e2h_matrix).float()

        # pn x en x 1 x 1.
        self.register_buffer('weight', e2h_matrix.view(
            e2h_matrix.size(0), e2h_matrix.size(1), 1, 1))

        # some keypoints are not coverred by any edges,
        # in these cases, we must add a constant bias to their heatmap weights.
        bias = ((e2h_matrix @ torch.ones(e2h_matrix.size(1)).to(
            e2h_matrix)) < 0.5).to(e2h_matrix)
        # pn x 1.
        self.register_buffer('bias', bias)

    def forward(self, edgemaps):
        # input: batch_size x en x hw x hh.
        # output: batch_size x pn x hw x hh.
        return F.conv2d(edgemaps, weight=self.weight, bias=self.bias)
    
class ConvBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, groups=1):
        super(ConvBlock, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size,
                              stride, padding=(kernel_size - 1) // 2, groups=groups, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Activation(nn.Module):
    def __init__(self, kind: str = 'relu', channel=None):
        super().__init__()
        self.kind = kind

        if '+' in kind:
            norm_str, act_str = kind.split('+')
        else:
            norm_str, act_str = 'none', kind

        self.norm_fn = {
            'in': F.instance_norm,
            'bn': nn.BatchNorm2d(channel),
            'bn_noaffine': nn.BatchNorm2d(channel, affine=False, track_running_stats=True),
            'none': None
        }[norm_str]

        self.act_fn = {
            'relu': F.relu,
            'softplus': nn.Softplus(),
            'exp': torch.exp,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'none': None
        }[act_str]

        self.channel = channel

    def forward(self, x):
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def extra_repr(self):
        return f'kind={self.kind}, channel={self.channel}'
    
class decoder_default:
    def __init__(self, weight=1, use_weight_map=False):
        self.weight = weight
        self.use_weight_map = use_weight_map

    def _make_grid(self, h, w):
        yy, xx = torch.meshgrid(
            torch.arange(h).float() / (h - 1) * 2 - 1,
            torch.arange(w).float() / (w - 1) * 2 - 1)
        return yy, xx

    def my_get_coords_from_heatmap(self, heatmap):
        batch, npoints, h, w = heatmap.shape
        yy, xx = self._make_grid(h, w)
        yy = yy.view(1, 1, h, w).to(heatmap)
        xx = xx.view(1, 1, h, w).to(heatmap)
        
        yy_coord = (yy * heatmap).sum([2, 3]) 
        xx_coord = (xx * heatmap).sum([2, 3]) 
        coords = torch.stack([xx_coord, yy_coord], dim=-1)

        return coords
    def get_coords_from_heatmap(self, heatmap):
        """
            inputs:
            - heatmap: batch x npoints x h x w

            outputs:
            - coords: batch x npoints x 2 (x,y), [-1, +1]
            - radius_sq: batch x npoints
        """
        batch, npoints, h, w = heatmap.shape
        if self.use_weight_map:
            heatmap = heatmap * self.weight

        yy, xx = self._make_grid(h, w)
        yy = yy.view(1, 1, h, w).to(heatmap)
        xx = xx.view(1, 1, h, w).to(heatmap)

        heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

        yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
        xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
        coords = torch.stack([xx_coord, yy_coord], dim=-1)

        return coords


EfficientFormer_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],  # 26m 83.3%
    'S2': [4, 4, 12, 8],  # 12m
    'S1': [3, 3, 9, 6],  # 79.0
    'S0': [2, 2, 6, 4],  # 75.7
}

# 26m
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

# 12m
expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

# 6.1m
expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}

# 3.5m
expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}


class Attention4D(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=8,
                 act_layer=nn.ReLU,
                 stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                                             nn.BatchNorm2d(dim), )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution ** 2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.q = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2d(self.num_heads * self.key_dim), )
        self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2d(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.d, 1),
                               nn.BatchNorm2d(self.num_heads * self.d),
                               )
        self.v_local = nn.Sequential(nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                                               kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
                                     nn.BatchNorm2d(self.num_heads * self.d), )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(act_layer(),
                                  nn.Conv2d(self.dh, dim, 1),
                                  nn.BatchNorm2d(dim), )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = (
                (q @ k) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        # attn = (q @ k) * self.scale
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v)

        out = x.transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution) + v_local
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out


def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        act_layer(),
    )


class LGQuery(torch.nn.Module):
    def __init__(self, in_dim, out_dim, resolution1, resolution2):
        super().__init__()
        self.resolution1 = resolution1
        self.resolution2 = resolution2
        self.pool = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),
                                   )
        self.proj = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1),
                                  nn.BatchNorm2d(out_dim), )

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention4DDownsample(torch.nn.Module):
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4,
                 resolution=8,
                 out_dim=None,
                 act_layer=None,
                 ):
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        self.resolution = resolution

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = dim
        self.resolution2 = math.ceil(self.resolution / 2)
        self.q = LGQuery(dim, self.num_heads * self.key_dim, self.resolution, self.resolution2)

        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2

        self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2d(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.d, 1),
                               nn.BatchNorm2d(self.num_heads * self.d),
                               )
        self.v_local = nn.Sequential(nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                                               kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d),
                                     nn.BatchNorm2d(self.num_heads * self.d), )

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim), )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(
            range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = (
                (q @ k) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )

        # attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 3)
        out = x.reshape(B, self.dh, self.resolution2, self.resolution2) + v_local

        out = self.proj(out)
        return out


class Embedding(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                 light=False, asub=False, resolution=None, act_layer=nn.ReLU, attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans),
                nn.BatchNorm2d(in_chans),
                nn.Hardswish(),
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(embed_dim)
            )
        elif self.asub:
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim,
                                   resolution=resolution, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class AttnFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=8, stride=None):

        super().__init__()

        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))

        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class FFN(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


def eformer_block(dim, index, layers,
                  pool_size=3, mlp_ratio=4.,
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                  drop_rate=.0, drop_path_rate=0.,
                  use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1, resolution=8, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            if index == 2:
                stride = 2
            else:
                stride = None
            blocks.append(AttnFFN(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution,
                stride=stride,
            ))
        else:
            blocks.append(FFN(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormerV2(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 pool_size=3,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=0,
                 distillation=True,
                 resolution=256,
                 e_ratios=expansion_ratios_L,
                 edge_info=None,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.decoder = decoder_default()
        self.patch_embed = stem(3, embed_dims[0], act_layer=act_layer)
        self.heatmap_act = Activation("in+relu", 51)
        
        network = []
        for i in range(len(layers)):
            stage = eformer_block(embed_dims[i], i, layers,
                                  pool_size=pool_size, mlp_ratio=mlp_ratios,
                                  act_layer=act_layer, norm_layer=norm_layer,
                                  drop_rate=drop_rate,
                                  drop_path_rate=drop_path_rate,
                                  use_layer_scale=use_layer_scale,
                                  layer_scale_init_value=layer_scale_init_value,
                                  resolution=math.ceil(resolution / (2 ** (i + 2))),
                                  vit_num=vit_num,
                                  e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                if i >= 2:
                    asub = True
                else:
                    asub = False
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                        resolution=math.ceil(resolution / (2 ** (i + 2))),
                        asub=asub,
                        act_layer=act_layer, norm_layer=norm_layer,
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()
        self.final_upsample1 = nn.Sequential(nn.ConvTranspose2d(embed_dims[-1], embed_dims[-1], 6, stride=4, padding=1),nn.BatchNorm2d(embed_dims[-1]), nn.ReLU())
        self.final_upsample2 = nn.Sequential(nn.ConvTranspose2d(embed_dims[-1], embed_dims[-1], 4, stride=2, padding=1),nn.BatchNorm2d(embed_dims[-1]), nn.ReLU())
        self.out_pointmaps = ConvBlock(embed_dims[-1], 51, 1, relu=False, bn=False)
        self.out_edgemaps = ConvBlock(embed_dims[-1], 8, 1, relu=False, bn=False)
        self.out_heatmaps = ConvBlock(embed_dims[-1], 51, 1, relu=False, bn=False)
        self.pointmap_act = Activation("sigmoid", 51)
        self.edgemap_act = Activation("sigmoid", 8)
        self.e2h_transform = E2HTransform(edge_info, 51, 8)
    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        #x = x.to('cuda')
        y, fusionmaps = [], []
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        # print(x.size())
        # goal: 32, 51, 64, 64
        x = self.norm(x)
        x = self.final_upsample1(x)
        x = self.final_upsample2(x)
        #x = self.upsample(x)
        # x : 16 384 8 8
        #heatmaps0 = self.final_upsample(x)
        
        heatmaps = self.heatmap_act(self.out_heatmaps(x))
        
        pointmaps0 = self.out_pointmaps(x)
        pointmaps = self.pointmap_act(pointmaps0)
        edgemaps0 = self.out_edgemaps(x)
        edgemaps = self.edgemap_act(edgemaps0)
        mask = self.e2h_transform(edgemaps) * pointmaps
        fusion_heatmaps = mask * heatmaps
        
        
        landmarks = self.decoder.get_coords_from_heatmap(fusion_heatmaps)
        y.append(landmarks)
            
        y.append(pointmaps)
        y.append(edgemaps)

        fusionmaps.append(fusion_heatmaps)
        # if self.dist:
        #     cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
        #     if not self.training:
        #         cls_out = (cls_out[0] + cls_out[1]) / 2
        # else:
        #     cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return y, fusionmaps, landmarks


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


@register_model
def efficientformerv2_s0(pretrained=False, edge_info=None,**kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S0'],
        embed_dims=EfficientFormer_width['S0'],
        downsamples=[True, True, True, True, True],
        vit_num=2,
        drop_path_rate=0.0,
        e_ratios=expansion_ratios_S0,
        edge_info=edge_info,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        checkpoint_model = torch.load('/disk2/icml/STAR/checkpoints/eformer_s0_450.pth')['model']
        for k in model.state_dict().keys():
                if k in checkpoint_model and checkpoint_model[k].shape != model.state_dict()[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        model.load_state_dict(checkpoint_model, strict=False)
    return model


@register_model
def efficientformerv2_s1(pretrained=False, edge_info=None, **kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S1'],
        embed_dims=EfficientFormer_width['S1'],
        downsamples=[True, True, True, True],
        vit_num=2,
        drop_path_rate=0.0,
        e_ratios=expansion_ratios_S1,
        edge_info=edge_info,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def efficientformerv2_s2(pretrained=False, **kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['S2'],
        embed_dims=EfficientFormer_width['S2'],
        downsamples=[True, True, True, True],
        vit_num=4,
        drop_path_rate=0.02,
        e_ratios=expansion_ratios_S2,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def efficientformerv2_l(pretrained=False, edge_info=None, **kwargs):
    model = EfficientFormerV2(
        layers=EfficientFormer_depth['L'],
        embed_dims=EfficientFormer_width['L'],
        downsamples=[True, True, True, True],
        vit_num=6,
        drop_path_rate=0.1,
        e_ratios=expansion_ratios_L,
        edge_info=edge_info,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        checkpoint_model = torch.load('/disk2/icml/STAR/checkpoints/eformer_l_450.pth')['model']
        for k in model.state_dict().keys():
                if k in checkpoint_model and checkpoint_model[k].shape != model.state_dict()[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        model.load_state_dict(checkpoint_model, strict=False)
    return model



def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


edge_info = (
                (True, (0, 1, 2, 3, 4)),  # RightEyebrow
                (True, (5, 6, 7, 8, 9)),  # LeftEyebrow
                (False, (10, 11, 12, 13)),  # NoseLine
                (False, (14, 15, 16, 17, 18)),  # Nose
                (True, (19, 20, 21, 22, 23, 24)),  # RightEye
                (True, (25, 26, 27, 28, 29, 30)),  # LeftEye
                (True, (31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)),  # OuterLip
                (True, (43, 44, 45, 46, 47, 48, 49, 50)),  # InnerLip
            )

# model = efficientformerv2_s0(pretrained=True, edge_info=edge_info)
# total_params = sum(
# 	param.numel() for param in model.parameters()
# )
# print(total_params)
#x = model(torch.zeros((1,3,256,256)))
#print(x)
#getModelSize(model)
#stat(model, (3, 256, 256))


# model = efficientformerv2_s0(pretrained=True, edge_info=edge_info)
# stat(model, (3, 256, 256))
# device = torch.device("cuda")
# model.to(device)
# dummy_input = torch.randn(1, 3,256,256, dtype=torch.float).to(device)
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# repetitions = 300
# timings=np.zeros((repetitions,1))
# #GPU-WARM-UP
# for _ in range(10):
#     _ = model(dummy_input)
# # MEASURE PERFORMANCE
# with torch.no_grad():
#     for rep in range(repetitions):
#         starter.record()
#         _ = model(dummy_input)
#         ender.record()
#         # WAIT FOR GPU SYNC
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time(ender)
#         timings[rep] = curr_time

# mean_syn = np.sum(timings) / repetitions
# std_syn = np.std(timings)
# print(mean_syn)



# ===========================================
# Total params: 3,241,136
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Total memory: 66.94MB
# Total MAdd: 1.01GMAdd
# Total Flops: 512.0MFlops
# Total MemR+W: 111.54MB