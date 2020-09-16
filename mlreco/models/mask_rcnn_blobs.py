# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Construct minibatches for Mask R-CNN training. Handles the minibatch blobs
that are specific to Mask R-CNN. Other blobs that are generic to RPN or
Fast/er R-CNN are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

import mlreco.models.utils.blob as blob_utils
import mlreco.models.utils.boxes as box_utils
import mlreco.models.utils.segms as segm_utils

from scipy.ndimage import zoom

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#larcvdataset original imports
import os,time
import ROOT
# from larcv import larcv
import numpy as np
from torch.utils.data import Dataset
#new imports:
try:
    import cv2
    import torchvision
except:
    pass

def add_mask_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add Mask R-CNN specific blobs to the input blob dictionary."""
    # Prepare the mask targets by associating one gt mask to each training roi
    # that has a fg (non-bg) class label.
#     print("roidb['id']      :     ", roidb['id'])
#     M = cfg.MRCNN.RESOLUTION
    M = 192
    
    polys_gt_inds = np.where((roidb['gt_classes'] > 0))[0]

    # input mask instead of polygon
#     clustermask_cluster_crop_chain = roidb['chain_cluster']
#     clustermask_cluster_crop_chain.GetEntry(roidb['id'])
#     entry_clustermaskcluster_crop_data = clustermask_cluster_crop_chain.clustermask_masks_branch
#     clustermaskcluster_crop_array = entry_clustermaskcluster_crop_data.as_vector()
#     cluster_masks = clustermaskcluster_crop_array[roidb['plane']]
#     masks_orig_size = []
#     boxes_from_polys = np.empty((0,4))
#     for i in polys_gt_inds:

#         bbox = larcv.as_ndarray_bbox(cluster_masks[int(i)])
#         # print(boxes_from_polys.shape)
#         # print(np.array([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]).shape)
#         boxes_from_polys = np.append(boxes_from_polys, np.array([[bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]]), 0)
#         # print(i)
#         # print(bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])
#         masks_orig_size.append(np.transpose(larcv.as_ndarray_mask(cluster_masks[int(i)])))




    # polys_gt = [roidb['segms'][i] for i in polys_gt_inds]
    # for i in range(len(polys_gt)):
    #     # print("i is:", i)
    #     poly = polys_gt[i]
    #     if len(poly) ==0:
    #         print()
    #         print('Cheated, and made my own box')
    #         print()
    #         poly = [[0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 0.0]]
    #         polys_gt[i] = poly
    # print('Type Boxes: ', type(boxes_from_polys))
    # print('Shape Boxes: ', boxes_from_polys.shape)
    # print('Boxes: ', boxes_from_polys)
    # boxes_from_polys = segm_utils.polys_to_boxes(polys_gt)
    # print('Type Boxes: ', type(boxes_from_polys))
    #
    # print('Shape Boxes: ', boxes_from_polys.shape)
    # print('Boxes: ', boxes_from_polys)
    
    
    gt_boxes = roidb['boxes'][:len(roidb['segms']), :]
    gt_boxes[:, 3:] -= 1
    
    boxes_from_polys = gt_boxes
    
    masks_orig_size = []
    
    # flatten segments
    im_coords = np.array([j for sub in roidb['segms'] for j in sub], dtype=np.int32)
    
#     im = np.zeros((1024, 1024, 1024), dtype=np.uint8)
    
#     for i in range(len(im_coords)):
#         im[im_coords[i][0]][im_coords[i][1]][im_coords[i][2]] = 1
        
    for box_ind in range(len(gt_boxes)):
        
        width = int(gt_boxes[box_ind][3] - gt_boxes[box_ind][0]) + 1
        height = int(gt_boxes[box_ind][4] - gt_boxes[box_ind][1]) + 1
        depth = int(gt_boxes[box_ind][5] - gt_boxes[box_ind][2]) + 1
        
        mask = np.zeros((width + 1, height + 1, depth + 1))
        
        coords = roidb['segms'][box_ind].astype(np.int32)
        
#         x_off = 2
#         y_off = 2
#         z_off = 2
        
#         if gt_boxes[box_ind][0] == 0:
#             x_off = 1
#         elif gt_boxes[box_ind][1] == 0:
#             y_off = 1
#         elif gt_boxes[box_ind][2] == 0:
#             z_off = 1
#         mask += im[int(np.ceil(gt_boxes[box_ind][0])):int(np.ceil(gt_boxes[box_ind][3])) + x_off, int(np.ceil(gt_boxes[box_ind][1])):int(np.ceil(gt_boxes[box_ind][4])) + y_off, int(np.ceil(gt_boxes[box_ind][2])):int(np.ceil(gt_boxes[box_ind][5])) + z_off]
        
        for i in range(len(im_coords)):
            if im_coords[i][0] >= gt_boxes[box_ind][0] and im_coords[i][0] < gt_boxes[box_ind][0] + width and im_coords[i][1] >= gt_boxes[box_ind][1] and im_coords[i][1] < gt_boxes[box_ind][1] + height and im_coords[i][2] >= gt_boxes[box_ind][2] and im_coords[i][2] < gt_boxes[box_ind][2] + depth:
                mask[int(im_coords[i][0] - gt_boxes[box_ind][0])][int(im_coords[i][1] - gt_boxes[box_ind][1])][int(im_coords[i][2] - gt_boxes[box_ind][2])] = 1
                
        for i in range(len(coords)):
            mask[coords[i][0] - np.min(coords[:,0])][coords[i][1] - np.min(coords[:,1])][coords[i][2] - np.min(coords[:,2])] = 2
        
        masks_orig_size.append(mask)

    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = blobs['labels_int32'].copy()
    roi_has_mask[roi_has_mask > 0] = 1

    if fg_inds.shape[0] > 0:
        # Class labels for the foreground rois
        mask_class_labels = blobs['labels_int32'][fg_inds]
        masks = blob_utils.zeros((fg_inds.shape[0], M**3), int32=True)



        # Find overlap between all foreground rois and the bounding boxes
        # enclosing each segmentation
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False))
        # Map from each fg rois to the index of the mask with highest overlap
        # (measured by bbox overlap)
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # original = np.array([[1,2,3], [4,5,6], [7,8,9]])
        # print('original[0][2]', original[0][2])
        # box_orig = np.array([7,7,9,9])
        # box_adjust = np.array([2,2,6,6])
        # resized = resize_mask_to_set_dim(original, box_adjust, box_orig, 10)

        # add fg targets
        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            # poly_gt = polys_gt[fg_polys_ind]
            mask_gt_orig_size = masks_orig_size[fg_polys_ind]
            box_gt = boxes_from_polys[fg_polys_ind]


            roi_fg = rois_fg[i]
            # Rasterize the portion of the polygon mask within the given fg roi
            # to an M x M binary image

            # mask = segm_utils.polys_to_mask_wrt_box(poly_gt, roi_fg, M)
            mask = resize_mask_to_set_dim(mask_gt_orig_size, roi_fg, box_gt, M)
#             mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
            mask = np.array(mask, dtype=np.int32)
            masks[i, :] = np.reshape(mask, M**3)
    else:  # If there are no fg masks (it does happen)

        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        # rois_fg is actually one background roi, but that's ok because ...
        rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # We give it an -1's blob (ignore label)
        masks = -blob_utils.ones((1, M**3), int32=True)
        # We label it with class = 0 (background)
        mask_class_labels = blob_utils.zeros((1, ))
        # Mark that the first roi has a mask
        roi_has_mask[0] = 1
    
    masks = _expand_to_class_specific_mask_targets(masks, mask_class_labels)

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))

    # Update blobs dict with Mask R-CNN blobs
    blobs['mask_rois'] = rois_fg
    blobs['roi_has_mask_int32'] = roi_has_mask
    blobs['masks_int32'] = masks


def _expand_to_class_specific_mask_targets(masks, mask_class_labels):
    """Expand masks from shape (#masks, M ** 2) to (#masks, #classes * M ** 2)
    to encode class specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]
    M = 192

    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -blob_utils.ones(
        (masks.shape[0], 6 * M**3), int32=True)

    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = M**3 * cls
        end = start + M**3
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]

    return mask_targets

def resize_mask_to_set_dim(mask_gt_orig_size, roi_fg, box_gt, M):
    """Take in original binary gt mask at original size in gt bbox
    Then take roi_fg (the predicted bbox) and crop the gt mask into it
    Finally output a square matrix pooled or upsampled version to
    dimensions MxM
    """
    #plus one to include the
    pred_w = int(roi_fg[3] - roi_fg[0] + 1)
    pred_h = int(roi_fg[4] - roi_fg[1] + 1)
    pred_d = int(roi_fg[5] - roi_fg[2] + 1)

    mask_cropped = np.zeros((pred_w, pred_h, pred_d, 1), dtype=np.uint8)

    #Find x indices to copy
    if box_gt[0] >= roi_fg[0]:
        start_copy_x = int(box_gt[0] - roi_fg[0])
    else:
        start_copy_x = 0
    if box_gt[3] >= roi_fg[3]:
        end_copy_x = pred_w
    else:
        end_copy_x = int(box_gt[3] - roi_fg[0] + 1)

    #Find y indices to copy
    if box_gt[1] >= roi_fg[1]:
        start_copy_y = int(box_gt[1] - roi_fg[1])
    else:
        start_copy_y = 0
    if box_gt[4] >= roi_fg[4]:
        end_copy_y = pred_h
    else:
        end_copy_y = int(box_gt[4] - roi_fg[1] + 1)
        
    #Find z indices to copy
    if box_gt[2] >= roi_fg[2]:
        start_copy_z = int(box_gt[2] - roi_fg[2])
    else:
        start_copy_z = 0
    if box_gt[5] >= roi_fg[5]:
        end_copy_z = pred_d
    else:
        end_copy_z = int(box_gt[5] - roi_fg[2] + 1)

#     for x in range(start_copy_x, end_copy_x):
#         for y in range(start_copy_y, end_copy_y):
#             for z in range(start_copy_z, end_copy_z):
#                 mask_cropped[x][y][z] = np.uint8(mask_gt_orig_size[ x - int(box_gt[0] - roi_fg[0]) ][ y - int(box_gt[1] - roi_fg[1])][ z - int(box_gt[2] - roi_fg[2]) ])
    
    x_off = int(box_gt[0] - roi_fg[0])
    y_off = int(box_gt[1] - roi_fg[1])
    z_off = int(box_gt[2] - roi_fg[2])
    
    gt_reshape = np.uint8(mask_gt_orig_size[start_copy_x - x_off:end_copy_x - x_off, start_copy_y - y_off:end_copy_y - y_off, start_copy_z - z_off:end_copy_z - z_off])
    
    shape = gt_reshape.shape
    
    mask_cropped[0:shape[0], 0:shape[1], 0:shape[2], 0] = gt_reshape
    
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.voxels(mask_cropped[:, :, :, 0], edgecolor="k")
#     plt.savefig("./masks/mask_before_%d" % int(roi_fg.sum()))
    
    #now we need to figure out how to resize this to a constant MxM
    mask_resized = zoom(mask_cropped[:, :, :, 0], (M / mask_cropped.shape[0], M / mask_cropped.shape[1], M / mask_cropped.shape[2]), order=0)
    
#     import pdb; pdb.set_trace()
#     mask_resized[mask_resized == 2] = 0
    
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.voxels(np.array(mask_resized), edgecolor="k")
#     plt.savefig("./masks/1mask_%d" % int(roi_fg.sum()))
    
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.voxels(np.array(mask_resized), edgecolor="k")
#     plt.savefig("./masks/1mask_%d" % int(roi_fg.sum()))
    
    return np.array(mask_resized)
