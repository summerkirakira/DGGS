#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def depth2image(depth_map):
    depth_map = depth_map.repeat(3, 1, 1)
    depth_map = (depth_map - torch.min(depth_map)) / (torch.max(depth_map) - torch.min(depth_map))
    depth_map[0, :, :] = 255 * (1 - depth_map[0, :, :])
    depth_map[1, :, :] = 0
    depth_map[2, :, :] = 255 * depth_map[2, :, :]
    return depth_map

def confidence2image(confidence_map):
    confidence_map = confidence_map.repeat(3, 1, 1)
    confidence_map = (confidence_map - torch.min(confidence_map)) / (torch.max(confidence_map) - torch.min(confidence_map)) * 255
    return confidence_map