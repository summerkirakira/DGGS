import torch
from typing import Tuple
from torch import Tensor
from PIL import Image
import numpy as np


model = None


def estimate_depth(image: Tensor, pcd, R, T, focal_x, focal_y) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """

    :param image: (B, 3, H, W)
    :return: pred_depth: (B, 1, H, W), confidence: (B, 1, H, W), normal: (B, 3, H, W), normal_confidence: (B, 1, H, W)
    """

    global model

    image = image.cuda()
    pred_depth, confidence, output_dict = model.inference({'input': image})

    height_start = (pred_depth.shape[2] - image.shape[2]) // 2
    height_end = height_start + image.shape[2]

    width_start = (pred_depth.shape[3] - image.shape[3]) // 2
    width_end = width_start + image.shape[3]

    pred_depth = pred_depth[:, :, height_start:height_end, width_start:width_end]
    confidence = confidence[:, :, height_start:height_end, width_start:width_end]

    normal = output_dict['prediction_normal'][:, :3, height_start:height_end, width_start:width_end]
    normal_confidence = output_dict['prediction_normal'][:, 3, height_start:height_end, width_start:width_end]

    height, width = image.shape[2], image.shape[3]
    sparse_depth, depth_weight = np.zeros((height, width)), np.zeros((height, width))

    K = np.array([[focal_x, 0, width / 2], [0, focal_y, height / 2], [0, 0, 1]])

    cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3, 1))

    valid_idx = np.where(np.logical_and.reduce(
        (cam_coord[2] >= 0,
         cam_coord[0] / cam_coord[2] >= 0, cam_coord[0] / cam_coord[2] <= width - 1,
         cam_coord[1] / cam_coord[2] >= 0, cam_coord[1] / cam_coord[2] <= height - 1)))[0]

    pts_depth = cam_coord[-1:, valid_idx]
    cam_coord = cam_coord[:2, valid_idx] / cam_coord[-1:, valid_idx]

    sparse_depth[np.round(cam_coord[1]).astype(np.int32),
    np.round(cam_coord[0]).astype(np.int32)] = pts_depth
    depth_weight[np.round(cam_coord[1]).astype(np.int32),
    np.round(cam_coord[0]).astype(np.int32)] = 1 / pcd.errors[valid_idx]
    depth_weight = depth_weight / depth_weight.max()

    # Image.fromarray(depth_weight).show()

    refined_depth, depth_loss = optimize_depth(source=pred_depth.squeeze().cpu().numpy(),
                                               target=sparse_depth,
                                               mask=sparse_depth > 0.0, depth_weight=depth_weight)


    # nomal_depth = (refined_depth - refined_depth.min()) / (refined_depth.max() - refined_depth.min())

    refined_depth = torch.from_numpy(refined_depth).cuda().unsqueeze(0).unsqueeze(0)
    depth_weight = torch.from_numpy(depth_weight).cuda()


    return refined_depth, confidence, normal, normal_confidence, depth_weight


def init_depth_model():
    global model
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True)
    model = model.cuda()


def remove_depth_model():
    global model
    model = None
    torch.cuda.empty_cache()


def optimize_depth(source, target, mask, depth_weight, prune_ratio=0.001):
    source = torch.from_numpy(source).cuda()  # np.array(h,w)
    target = torch.from_numpy(target).cuda()
    mask = torch.from_numpy(mask).cuda()
    depth_weight = torch.from_numpy(depth_weight).cuda()

    with torch.no_grad():
        target_depth_sorted = target[target > 1e-7].sort().values
        min_prune_threhold = target_depth_sorted[int(target_depth_sorted.numel() * prune_ratio)]
        max_prune_threhold = target_depth_sorted[int(target_depth_sorted.numel() * (1 - prune_ratio))]

        mask2 = target > min_prune_threhold
        mask3 = target < max_prune_threhold
        mask = torch.logical_and(mask, torch.logical_and(mask2, mask3))

    source_masked = source[mask]
    target_masked = target[mask]
    depth_weight_masked = depth_weight[mask]

    scale = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

    optimizer = torch.optim.Adam([scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8 ** (1 / 100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0

    while abs(loss_ema - loss_prev) > 1e-5:
        source_hat = scale * source_masked + shift
        loss = torch.mean(((target_masked - source_hat) ** 2) * depth_weight_masked)

        # penalize depths not in [0,1]
        loss_hinge1 = loss_hinge2 = 0.0
        if (source_hat <= 0.0).any():
            loss_hinge1 = 2.0 * ((source_hat[source_hat <= 0.0]) ** 2).mean()

        loss = loss + loss_hinge1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        iteration += 1
        if iteration % 2000 == 0:
            # print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    loss = loss.item()
    # print(f"Final loss={loss:10.5f}")

    with torch.no_grad():
        source_refined = (scale * source + shift).cpu().numpy()

    torch.cuda.empty_cache()
    return source_refined, loss

