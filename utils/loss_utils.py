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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

import numpy as np
import cv2
from pytorch3d.ops.knn import knn_points

def l1_loss(network_output, gt, mask=None):
    if mask is None:
        return torch.abs((network_output - gt)).mean()
    if mask.shape[0] == 1:
        mask = mask.repeat(network_output.shape[0], 1, 1)
    return (torch.abs((network_output - gt)) * mask) / mask.sum()

def l2_loss(network_output, gt, mask=None):
    if mask is None:
        return ((network_output - gt) ** 2).mean()
    if mask.shape[0] == 1:
        mask = mask.repeat(network_output.shape[0], 1, 1)
    return ((network_output - gt) ** 2 * mask) / mask.sum()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def full_aiap_loss(gs_can, gs_obs, n_neighbors=5):
    xyz_can = gs_can.get_xyz
    xyz_obs = gs_obs.get_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_obs.get_covariance()

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0),
                             xyz_can.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    loss_xyz = aiap_loss(xyz_can, xyz_obs, nn_ix=nn_ix)
    loss_cov = aiap_loss(cov_can, cov_obs, nn_ix=nn_ix)

    return loss_xyz, loss_cov

def aiap_loss(x_canonical, x_deformed, n_neighbors=5, nn_ix=None):
    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    if nn_ix is None:
        _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                                 x_canonical.unsqueeze(0),
                                 K=n_neighbors + 1,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0)

    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[nn_ix])[:,0,1:]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[nn_ix])[:,0,1:]

    loss = F.l1_loss(dists_canonical, dists_deformed)

    return loss

def tv_loss(pred, mask=None):
    """
    Total variation loss
    """
    if mask is None:
        if len(pred.shape) == 4:
            dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        else:
            dy = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
            dx = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        return torch.mean(dx) + torch.mean(dy)
    
    if mask.shape[0] == 1:
        mask = mask.repeat(pred.shape[0], 1, 1)
        
    if len(pred.shape) == 4:
        raise NotImplementedError
    else:
        # Handle the case for non-batch data (assuming (H, W) shape)
        dy = torch.abs(pred[1:, :] - pred[:-1, :]) * mask[1:, :] * mask[:-1, :]
        dx = torch.abs(pred[:, 1:] - pred[:, :-1]) * mask[:, 1:] * mask[:, :-1]

    # Only consider masked regions, prevent division by zero
    valid_dy = mask[:, 1:, :] * mask[:, :-1, :]
    valid_dx = mask[:, :, 1:] * mask[:, :, :-1]

    return (torch.sum(dx) / torch.clamp(torch.sum(valid_dx), min=1) +
            torch.sum(dy) / torch.clamp(torch.sum(valid_dy), min=1))

