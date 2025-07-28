# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type
from .SMoE import LinearExperts

# GELU激活函数：GELU 的特点在于它结合了 ReLU（Rectified Linear Unit）激活函数的线性和非线性部分。
# 对于正输入值，GELU 表现为线性函数，斜率为 1；对于负输入值，它逐渐减小到 0，但不是像 ReLU 那样突然截断。
# 这种平滑的过渡有助于避免梯度消失问题，并允许模型在训练过程中更好地保留信息。

# 多层感知机，用于构建神经网络
class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,  # 输入和输出的词嵌入的维度
            mlp_dim: int,  # MLP中间层的维度
            act: Type[nn.Module] = nn.GELU,  # 使用的激活函数
            lora_r=32,
            lora_alpha=16,
            lora_num=4,

    ) -> None:
        super().__init__()
        self.lin1 = LinearExperts(in_features=embedding_dim, out_features=mlp_dim, lora_num=lora_num,
                                  r=lora_r, lora_alpha=lora_alpha)  # 输入层到中间层
        self.lin2 = LinearExperts(in_features=mlp_dim, out_features=embedding_dim, lora_num=lora_num,
                                  r=lora_r, lora_alpha=lora_alpha)# 中间层到输出层
        self.act = act()  # 激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播：线性层 -> 激活函数 -> 线性层
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    # LN 多使用在有Transformer的结构中
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))  # 可学习的权重
        self.bias = nn.Parameter(torch.zeros(num_channels))  # 可学习的偏置
        self.eps = eps  # 一个小数，用于数值的稳定性

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)  # 计算输入数值x的均值
        s = (x - u).pow(2).mean(1, keepdim=True)  # 计算输入数值的方差
        x = (x - u) / torch.sqrt(s + self.eps)  # 对数值进行标准化
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用缩放和偏置
        return x
