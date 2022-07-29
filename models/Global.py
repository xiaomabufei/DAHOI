# ------------------------------------------------------------------------
# DAHOI
# Copyright (c) 2022 Shuailei Ma. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from math import ceil
def get_relative_position_index(q_windows, k_windows):
    """
    Args:
        q_windows: tuple (query_window_height, query_window_width)
        k_windows: tuple (key_window_height, key_window_width)

    Returns:
        relative_position_index: query_window_height*query_window_width, key_window_height*key_window_width
    """
    # get pair-wise relative position index for each token inside the window
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])
    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = torch.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh_q*Ww_q, Wh_k*Ww_k, 2
    relative_coords[:, :, 0] += k_windows[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += k_windows[1] - 1
    relative_coords[:, :, 0] *= (q_windows[1] + k_windows[1]) - 1
    relative_position_index = relative_coords.sum(-1)  #  Wh_q*Ww_q, Wh_k*Ww_k
    return relative_position_index
def parition(x, scale_H, scale_W):
    r""" parition.
       Args:
           in: B C H W
           out: num scale_H scale_W C
       """
    B, C, H, W = x.shape
    x = torch.nn.functional.unfold(x, kernel_size=(scale_H, scale_W), stride=(scale_H, scale_W))
    x = x.transpose(1, 2)
    x = x.contiguous().view(-1, C, scale_H, scale_W)
    x = rearrange(x, 'N C H W ->N H W C', C=C)
    return x
class sub_window_pooling(nn.Module):
    r""" sub pooling.
           Args:
               in: num scale_H scale_W C
               out: B H/scale_H W/scale_W C
           """
    def __init__(self, down_scale, dim):
        super().__init__()
        self.scale_H, self.scale_W = down_scale
        self.pool = torch.nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(self.scale_H, self.scale_W), stride=(self.scale_H, self.scale_W))
    def forward(self, x):
        H, W = x.shape[1], x.shape[2]
        x = rearrange(x, 'N H W C->N C H W') # number h w dim
        x = self.pool(x) # N C 1 1
        B = int(x.shape[0] / (H * W / self.scale_H / self.scale_W))
        x = x.view(B, H // self.scale_H, W // self.scale_W, self.scale_H//self.scale_H, self.scale_W//self.scale_W, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H // self.scale_H, W // self.scale_W, -1)
        return x
class Global_Attention(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input attention.
        down_scale (tuple[int]): down sacle ratio.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, down_scale, dim, num_heads=8,
                 attn_drop=0, proj_drop=0, qk_scale=None, qkv_bias=None):
        super().__init__()
        self.dim = dim
        self.nc = dim * num_heads
        self.down_scale = down_scale
        self.scale_H, self.scale_W = self.down_scale
        self.pool = sub_window_pooling(self.down_scale, dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dwc_rpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((ceil(self.H/self.scale_H)+self.scale_H) * (ceil(self.W/self.scale_W)+self.scale_W), num_heads))
        # relative_position_index = get_relative_position_index((self.scale_H, self.scale_W), (ceil(self.H/self.scale_H), ceil(self.W/self.scale_W)))
        # self.register_buffer("relative_position_index", relative_position_index)
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.Q = nn.Linear(dim, dim, bias=qkv_bias)
        self.KV = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        """
        Args:
            x: Input feature, tensor size B C H W.
            H, W: Spatial resolution of the input feature.
        """
        H, W = x.shape[-2], x.shape[-1]
        scale_H, scale_W = self.down_scale
        pad_l = pad_t = 0
        pad_r = (self.scale_W - W % self.scale_W) % self.scale_W
        pad_b = (self.scale_H - H % self.scale_H) % self.scale_H
        B= x.shape[0]
        C = x.shape[1]
        origin_x = x
        x = rearrange(x, 'B C H W -> B H W C')
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x = rearrange(x, 'B H W C -> B C H W')
        _, _2, Hp, Wp = x.shape
        x_parition = parition(x, scale_H, scale_W) #num scale_H, scale_W C
        pool_x = self.pool(x_parition).view(B, -1, C)# B H/S W/S C
        x = x_parition.view(-1, scale_H*scale_W, C) # num S*S C
        pool_x = pool_x.repeat(x_parition.shape[0]//pool_x.shape[0], 1, 1) #num H/S*W/S C
        q = self.Q(x).reshape(x.shape[0], x.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.KV(pool_x).reshape(pool_x.shape[0], pool_x.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 2, B_, 8, 3N, C//8
        k, v = kv[0], kv[1]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.scale_H * self.scale_W, ceil(self.H/self.scale_H)*ceil(self.W/self.scale_W), -1)
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = self.dwc_rpe(origin_x)
        attn = self.softmax(attn)
        # attn = attn.view(-1, self.num_heads, x.shape[1], pool_x.shape[1])
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(1, 2).reshape(B, C, Hp, Wp)[:, :, :H, :W]
        attn = attn + relative_position_bias
        x = self.proj(attn.reshape(B, -1, C))
        x = self.proj_drop(x)
        # x = x.view(B, Hp, Wp, C)[:, :H, :W, :]
        # x = rearrange(x, 'B H W C -> B (H W) C')
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class GIM(nn.Module):
    def __init__(self, down_scale, dim, num_heads=2,mlp_ratio=4,act_layer=nn.GELU,drop_path=0.,
                     attn_drop=0, proj_drop=0, norm_layer=nn.LayerNorm, drop=0., qk_scale=None, qkv_bias=None):
        super().__init__()
        self.attn = Global_Attention(down_scale, dim, num_heads=num_heads,
                 attn_drop=0, proj_drop=0, qk_scale=None, qkv_bias=None)
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size B C H W.
            H, W: Spatial resolution of the input feature.
        """
        B, C, H, W = x.shape
        x = rearrange(x, 'B C H W -> B (H W) C ')
        shortcut = x # B H*W C
        x = self.norm1(x)
        x = x.view(B, C, H, W)
        attan = self.attn(x) # B H*W C
        x = attan
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x

if __name__ == '__main__':
    GIM = GIM((4, 4), dim=4, num_heads=2)
    input = torch.rand(1, 4, 225, 225)
    output = GIM(input)
    print(output.size())