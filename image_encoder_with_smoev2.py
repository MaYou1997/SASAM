# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .SMoEv2 import LinearExperts
from typing import Optional, Tuple, Type
from loralib import MergedLinear
from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
# 这里的类定义了编码器的主干部分，并且可以在VITDet骨干网络上进行适配
class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            img_size: int = 40,
            patch_size: int = 1,
            in_chans: int = 1024,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 16,
            mlp_ratio: float = 4.0,
            out_chans: int = 1024,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes=[2,5,8,11],
    ) -> None:
        """
        Args:
            img_size (int): Input image size.  输入图像的大小
            patch_size (int): Patch size. 每个块的大小，patch
            in_chans (int): Number of input image channels. 输入图像的通道数量
            embed_dim (int): Patch embedding dimension. 块嵌入的维度
            depth (int): Depth of ViT. VIT网络的深度
            num_heads (int): Number of attention heads in each ViT block. 每个VIT块中注意力的头数
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. MLP隐藏维度与嵌入维度的比例
            norm_layer (nn.Module): Normalization layer. 归一化的层
            act_layer (nn.Module): Activation layer. 激活层
            use_abs_pos (bool): If True, use absolute positional embeddings. 如果为True，则使用绝对位置的嵌入
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map. 在注意力图中添加相对位置的嵌入
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters. 如果设置为True,则将相对位置的参数初始化为0
            window_size (int): Window size for window attention blocks. 窗口注意力块的窗口大小
            global_attn_indexes (list): Indexes for blocks using global attention. 使用全局注意力的块的索引列表
        """
        super().__init__()
        self.img_size = img_size  # 输入图像的大小
        self.out_chans = out_chans

        # 初始化图像块的嵌入层
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),  # 定义块的大小
            stride=(patch_size, patch_size),  # 步长和块的大小保持一致
            in_chans=in_chans,  # 设置输入图像通道数字
            embed_dim=embed_dim,  # 设置嵌入的维度 768
        )
        # frozen weights:
        # for name, param in self.patch_embed.named_parameters():
            # param.requires_grad = False

        # 初始化位置嵌入，默认为None
        self.pos_embed: Optional[nn.Parameter] = None
        # 这里表示使用绝对位置嵌入
        if use_abs_pos:
            # 使用与训练的图像大小初始化绝对位置的嵌入
            # 这里的绝对位置嵌入，应该是用图像的大小除以patch的大小
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
        # frozen weights:
        # self.pos_embed.requires_grad = False
        # 开始叠加Transformer的块
        self.blocks = nn.ModuleList()
        for i in range(depth):  # 根据深度对块进行叠加
            block = Block(
                dim=embed_dim,  # 嵌入的维度
                num_heads=num_heads,  # 注意力头的数量
                mlp_ratio=mlp_ratio,  # MLP隐藏的维度和嵌入维度的比例
                qkv_bias=qkv_bias,  # 是否给查询、键和值添加偏置
                norm_layer=norm_layer,  # 归一化层
                act_layer=act_layer,  # 激活层
                use_rel_pos=use_rel_pos,  # 是否使用相对位置的嵌入
                rel_pos_zero_init=rel_pos_zero_init,  # 是否将相对位置的参数初始化为零
                window_size=window_size if i not in global_attn_indexes else 0,  # 窗口的大小
                input_size=(img_size // patch_size, img_size // patch_size),  # 输入的大小
            )
            self.blocks.append(block)
        self.multi_scale_1 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3,
                                       dilation=5, padding=5)
        self.multi_scale_2 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, 
                                       dilation=7, padding=7)
        self.multi_scale_3 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, 
                                       dilation=3, padding=3)
        self.routernet = nn.Linear(in_features=embed_dim, out_features=4, bias=False)

            # frozen the weight:
        # for name, param in self.blocks.named_parameters():
            # if not any(nd in name for nd in ['lora', 'mlp']):
                # param.requires_grad = False

        # 初始化模型的neck部分，用于进一步对Transformer的输出进行处理
        self.neck = nn.Sequential(
            # 卷积层
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            # LN层
            LayerNorm2d(out_chans),
            # 卷积层
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            # LN层
            LayerNorm2d(out_chans),
        )

        # for name, param in self.neck.named_parameters():
            # param.requires_grad = False

        # frozen the weight:
        # for name, param in self.blocks.named_parameters():
            # if not any(nd in name for nd in ['lora', 'mlp', 'bias']):
                # param.requires_grad = False

    # 这里主要拿到上面的块之后进行前向传播，定义前向传播的流程
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ouput_list = []
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # print(x.shape)
            # print(self.pos_embed.shape)
            x = x + self.pos_embed  # B, H, W, C

        for blk in self.blocks:

            #x = x.permute(0, 3, 1, 2)  # B, C, H, W
            #x_1 = self.multi_scale_1(x)
            #x_2 = self.multi_scale_2(x)
            #x_3 = self.multi_scale_3(x)
            #x = x + x_1 + x_2 + x_3
            #x = x.permute(0, 2, 3, 1)  # B, H, W, C

            x = blk(x)
            ouput_list.append(x)
        
        x_stage1 = ouput_list[2]
        x_stage2 = ouput_list[5]
        x_stage3 = ouput_list[8]
        x_stage4 = ouput_list[11]
        
        switch_list = []
        switch_list.append(x_stage1)
        switch_list.append(x_stage2)
        switch_list.append(x_stage3)
        switch_list.append(x_stage4)

        route_weight = nn.functional.softmax(self.routernet(x), dim=-1, dtype=torch.float32).to(x.dtype)
        
        for i in range(4):
            if len(x.shape) == 4:
                x = x + torch.unsqueeze(route_weight[:, :, :, i], -1) * switch_list[i]
            if len(x.shape) == 3:
                x = x + torch.unsqueeze(route_weight[:, :, i], -1) * switch_list[i]
        
        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""
    # VIT中Transformer中的基础模块， “支持窗口注意力和残差传播块的Transformer块”
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    # 带有相对注意力机制的多头自注意力
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
            lora_r=32,
            lora_alpha=16,
            lora_num=4
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.lora_num = lora_num
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # self.qkv = MergedLinear(dim, dim * 3, bias=qkv_bias, r=self.lora_r, enable_lora=[True, True, True])
        # self.qkv = nn.Linear(dim, dim * 3)
        self.qkv = LinearExperts(in_features=dim, out_features=dim * 3, r=self.lora_r, lora_num=self.lora_num,
                                lora_alpha=self.lora_alpha)
        # use usual linear and use sam weight
        # self.q_smoe = LinearExperts(dim // num_heads, dim // num_heads, bias=qkv_bias, r=self.lora_r, lora_alpha=self.lora_alpha)
        # self.q_smoe.weight.data = torch.eye(dim // num_heads, dim // num_heads)
        # self.k_smoe = LinearExperts(dim // num_heads, dim // num_heads, r=self.lora_r, lora_alpha=self.lora_alpha)
        # self.k_smoe.weight.data = torch.eye(dim // num_heads, dim // num_heads)
        # self.v_smoe = LinearExperts(dim // num_heads, dim // num_heads, r=self.lora_r, lora_alpha=self.lora_alpha)
        # self.v_smoe.weight.data = torch.eye(dim // num_heads, dim // num_heads)

        self.proj = LinearExperts(in_features=dim, out_features=dim, r=self.lora_r, lora_num=self.lora_num,
                                lora_alpha=self.lora_alpha)


        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        # q = self.q_smoe(q)

        # for i in range(self.q_smoe.lora_num):
            # weight = getattr(self.q_smoe, f"lora_A{i}").weight
            # getattr(self.k_smoe, f"lora_A{i}").weight = nn.Parameter(weight.clone())
            # getattr(self.v_smoe, f"lora_A{i}").weight = nn.Parameter(weight.clone())

        # k = self.k_smoe(k)
        # v = self.v_smoe(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    将数据分割成有填充的非重叠的窗口
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    将窗口中的数据还原为原始的序列，并移除填充的部分
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    根据查询和键的大小之间的相对位置，获取相对位置嵌入。
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    分解相对位置的嵌入
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    将图像转化为patch
    """
    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
