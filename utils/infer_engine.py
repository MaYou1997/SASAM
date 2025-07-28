import argparse
import os
import torch
import sys
from models.models_mae import *
from torch.nn import functional as F
import time

def define_embed_patches_idx(side_patches_num, centeral_patches_num):
    '''params:
    side_patches_num: the number of patches of per side of FOV,
    centeral_patches_num: the number of saved patches in the centeral of FOV

    target: get the given numbers patches of the centeral of FOV patches'''

    total_patches_num = int( side_patches_num ** 2)
    flatten_idx = torch.arange(total_patches_num)
    reshape_idx = flatten_idx.reshape(side_patches_num, side_patches_num) # reshape the shape same as patchified FOV

    mid_point = int ( side_patches_num // 2)
    mid_centeral_point = int ( centeral_patches_num // 2)

    selected_idx = reshape_idx[
        mid_point - mid_centeral_point : mid_point + mid_centeral_point,
        mid_point - mid_centeral_point : mid_point + mid_centeral_point
    ]

    return selected_idx.flatten()

# test1:
# side_patches_num = 16
# centeral_patches_num = 4
# selected_idx = define_embed_patches_idx(side_patches_num, centeral_patches_num)
# print(selected_idx)

def define_idx(centreal_patch_num, pix_size):
    index_h = torch.zeros(centreal_patch_num * centreal_patch_num)
    index_w = torch.zeros(centreal_patch_num * centreal_patch_num)
    count = 0

    for res_h in range(centreal_patch_num):
        for res_w in range(centreal_patch_num):
            index_h[count] = res_h * pix_size
            index_w[count] = res_w * pix_size
            count +=1
    return index_h, index_w

# test 2
# centreal_patches_num = 4
# pix_size = 5
# index_h, index_w = define_idx(centreal_patches_num, pix_size)
# print(index_h, index_w)

def place_res(res_stack, target_tensor, anchor_h, anchor_w, index_d, index_h, index_w, embed_idx):
    '''params:
    res_stack: the output of model.infer_latent
    target_tensor: ?
    anchor_h:
    anchor_w'''

    res_stack = res_stack[:, 1:][:, embed_idx.long()] # res_stack.shape = B, L, D. [:, 1:]->B,L-1,D. [:, embed_idx.long]. get the selected patches
    col_num = res_stack.shape[0]

    h = index_h + anchor_h
    h = h.repeat(col_num, 1)

    w = index_w + anchor_w
    w = w.repeat(col_num, 1)

    d = index_d.repeat_interleave(res_stack.shape[1])
    d = d.reshape(col_num, -1)

    target_tensor[d.long(), h.long(), w.long(), :] = res_stack.cpu()

def split(a,n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min( i + 1, m )] for i in range(n))

def run_inference(
        rank,
        ngpus_per_node,
        scr,
        cfg,
        model,
        embedding_storage_path,
        device
):
    orgD, orgH, orgW = scr.shape
    central_patch = cfg["MODEL"]["central_patch"] # 4
    target_size = cfg["DATASET"]["vol_size"] #80
    vol_size = (
        cfg["DATASET"]["vol_size"]
        + int(cfg["DATASET"]["vol_size"] // cfg["DATASET"]["patch_size"]
              if cfg["DATASET"]["patch_size"] % 2 == 0
              else cfg["DATASET"]["vol_size"])
    )

    pix_size = (
        cfg["DATASET"]["patch_size"] + 1
        if cfg["DATASET"]["patch_size"] % 2 == 0
        else cfg["DATASET"]["patch_size"]
    )