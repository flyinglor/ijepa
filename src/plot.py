# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from masks.multiblock_3d import MaskCollator as MBMaskCollator
from masks.utils import apply_masks

from datasets.custom_image_dataset import make_adni, make_ukb

from transforms import make_transforms, make_transforms_adni

import matplotlib.pyplot as plt

# --
log_timings = True
log_freq = 10

# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

def plot_3d(img, save_path="3dplot.png"):
    # Threshold to create binary data for voxels (0: empty, 1: filled)
    voxel_data = img > 0  # Example threshold

    # Create a 3D voxel plot
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Plot voxels
    ax.voxels(voxel_data, facecolors='blue', edgecolor='k', alpha=0.7)

    # Add labels and adjust view
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
        


def plot_and_save_images(imgs, contexts, targets1, targets2, targets3, targets4, save_path="output.png"):
    """
    Plot and save middle slices of imgs, contexts, and targets for visualization.

    Args:
        imgs (Tensor): Batch of input images, shape (B, D, H, W).
        contexts (Tensor): Batch of context images, shape (B, D, H, W).
        targets (Tensor): Batch of target images, shape (B, D, H, W).
        save_path (str): Path to save the resulting plot.
    """
    # Ensure dimensions are correct and find the middle slice index
    middle_index = 64  # Middle slice along depth

    batch_size = len(contexts)
    fig, axes = plt.subplots(batch_size, 6, figsize=(15, batch_size * 3))
    axes = np.atleast_2d(axes)

    for i in range(batch_size):
        ax = axes[i]

        # Plot middle slices
        ax[0].imshow(imgs[i, :, :, middle_index], cmap="gray")
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(contexts[i][:, :, middle_index], cmap="gray")
        ax[1].set_title("Context")
        ax[1].axis("off")

        ax[2].imshow(targets1[i][:, :, middle_index], cmap="gray")
        ax[2].set_title("Target-1")
        ax[2].axis("off")

        ax[3].imshow(targets2[i][:, :, middle_index], cmap="gray")
        ax[3].set_title("Target-2")
        ax[3].axis("off")

        ax[4].imshow(targets3[i][:, :, middle_index], cmap="gray")
        ax[4].set_title("Target-3")
        ax[4].axis("off")

        ax[5].imshow(targets4[i][:, :, middle_index], cmap="gray")
        ax[5].set_title("Target-4")
        ax[5].axis("off")

    # Adjust row spacing
    plt.subplots_adjust(hspace=0.1)  # Decrease hspace for less space between rows

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def reconstruct_image(img, excluded_patches, patch_size=16):
    """
    Reconstruct an image without displaying excluded patches.

    Args:
        img (torch.Tensor): The original image tensor (B, H, W, D).
        patch_size (int): Size of the patches (assumes cubic patches).
        excluded_patches (torch.Tensor): Tensor of patch indices to exclude.

    Returns:
        torch.Tensor: The reconstructed image with excluded patches blanked out.
    """
    H, W, D = img.shape
    reconstructed = torch.zeros_like(img)  # Initialize a blank image

    # Compute number of patches along each dimension
    N_H = H // patch_size
    N_W = W // patch_size
    N_D = D // patch_size

    patch_number = 0
    for x in range(N_H):
        for y in range(N_W):
            for z in range(N_D):
                if patch_number in excluded_patches:
                    # Copy the patch from the original image
                    reconstructed[
                        x * patch_size : (x + 1) * patch_size,
                        y * patch_size : (y + 1) * patch_size,
                        z * patch_size : (z + 1) * patch_size,
                    ] = img[
                        x * patch_size : (x + 1) * patch_size,
                        y * patch_size : (y + 1) * patch_size,
                        z * patch_size : (z + 1) * patch_size,
                    ]
                patch_number += 1

    return reconstructed

def main():
    crop_size = 128
    patch_size = 16
    pred_mask_scale = [0.015, 0.06]
    enc_mask_scale = [0.54, 0.67]
    aspect_ratio = [1.0, 1.0]
    num_enc_masks = 1
    num_pred_masks = 4
    allow_overlap = False
    min_keep = 1
    batch_size = 16
    pin_mem = True
    num_workers = 0
    world_size = 1
    rank = 0
    root_path = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/ADNI"
    image_folder = "imagenet_full_size/061417/"
    copy_data = False

    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    #adni transform
    transform = make_transforms_adni()

    #create adni dataloader
    _, unsupervised_loader, unsupervised_sampler = make_adni(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        drop_last=True,
        test=True)


    for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
        imgs = udata[0]
        # Remove the second dimension
        imgs = imgs.squeeze(1)
        B, H, W, D = imgs.shape

        contexts = []
        targets1 = []
        targets2 = []
        targets3 = []
        targets4 = []
        for i in range(11, 14):
            img = imgs[i]
            mask_enc = masks_enc[0][i]
            # print(mask_enc)
            masks_pred1 = masks_pred[0][i]
            # print(masks_pred1)
            masks_pred2 = masks_pred[1][i]
            masks_pred3 = masks_pred[2][i]
            masks_pred4 = masks_pred[3][i]
            #recon
            context = reconstruct_image(img, mask_enc)
            target1 = reconstruct_image(img, masks_pred1)
            target2 = reconstruct_image(img, masks_pred2)
            target3 = reconstruct_image(img, masks_pred3)
            target4 = reconstruct_image(img, masks_pred4)
            contexts.append(context)
            targets1.append(target1)
            targets2.append(target2)
            targets3.append(target3)
            targets4.append(target4)


        save_path = "batch_images_with_masks.png"
        # plot_3d(contexts[0])
        plot_and_save_images(imgs, contexts, targets1, targets2, targets3, targets4, save_path=save_path)
        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
