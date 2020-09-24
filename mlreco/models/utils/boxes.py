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
#
# Based on:
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Box manipulation functions. The internal Detectron box format is
[x1, y1, x2, y2] where (x1, y1) specify the top-left box corner and (x2, y2)
specify the bottom-right box corner. Boxes from external sources, e.g.,
datasets, may be in other formats (such as [x, y, w, h]) and require conversion.

This module uses a convention that may seem strange at first: the width of a box
is computed as x2 - x1 + 1 (likewise for height). The "+ 1" dates back to old
object detection days when the coordinates were integer pixel indices, rather
than floating point coordinates in a subpixel coordinate frame. A box with x2 =
x1 and y2 = y1 was taken to include a single pixel, having a width of 1, and
hence requiring the "+ 1". Now, most datasets will likely provide boxes with
floating point coordinates and the width should be more reasonably computed as
x2 - x1.

In practice, as long as a model is trained and tested with a consistent
convention either decision seems to be ok (at least in our experience on COCO).
Since we have a long history of training models with the "+ 1" convention, we
are reluctant to change it even if our modern tastes prefer not to use it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import numpy as np
import torch

#import mlreco.models.utils.cython_bbox as cython_bbox
#import mlreco.models.utils.cython_nms as cython_nms
import mlreco.models.nms.nms as c_nms

# bbox_overlaps = cython_bbox.bbox_overlaps

def bbox_overlaps_3D(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2, z1, z2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,6)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = boxes1.chunk(6, dim=1)
    b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = boxes2.chunk(6, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    z1 = torch.max(b1_z1, b2_z1)[:, 0]
    z2 = torch.min(b1_z2, b2_z2)[:, 0]
    zeros = torch.zeros(y1.size()[0], requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1 + 1, zeros) * torch.max(y2 - y1 + 1, zeros) * torch.max(z2 - z1 + 1, zeros)

    # 3. Compute unions
    b1_volume = (b1_y2 - b1_y1 + 1) * (b1_x2 - b1_x1 + 1)  * (b1_z2 - b1_z1 + 1)
    b2_volume = (b2_y2 - b2_y1 + 1) * (b2_x2 - b2_x1 + 1)  * (b2_z2 - b2_z1 + 1)
    union = b1_volume[:,0] + b2_volume[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)
    return overlaps

def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)]. / 3D: (z1, z2))
    For better performance, pass the largest set first and the smaller second.
    :return: (#boxes1, #boxes2), ious of each box of 1 machted with each of 2
    """

    # Areas of anchors and GT boxes
    volume1 = (boxes1[:, 3] - boxes1[:, 0] + 1) * (boxes1[:, 4] - boxes1[:, 1] + 1) * (boxes1[:, 5] - boxes1[:, 2] + 1)
    volume2 = (boxes2[:, 3] - boxes2[:, 0] + 1) * (boxes2[:, 4] - boxes2[:, 1] + 1) * (boxes2[:, 5] - boxes2[:, 2] + 1)
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes2.shape[0]):
        box2 = boxes2[i]  # this is the gt box
        overlaps[:, i] = compute_iou_3D(box2, boxes1, volume2[i], volume1)
    return overlaps

def compute_iou_3D(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2, z1, z2] (typically gt box)
    boxes: [boxes_count, (y1, x1, y2, x2, z1, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[4], boxes[:, 4])
    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[3], boxes[:, 3])
    z1 = np.maximum(box[2], boxes[:, 2])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1 + 1, 0) * np.maximum(y2 - y1 + 1, 0) * np.maximum(z2 - z1 + 1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    if union.any() == 0:
        print("0 union!")
        return 0
    iou = intersection / union

    return iou

def boxes_area(boxes):
    """Compute the area of an array of boxes."""
    w = (boxes[:, 2] - boxes[:, 0] + 1)
    h = (boxes[:, 3] - boxes[:, 1] + 1)
    areas = w * h

    neg_area_idx = np.where(areas < 0)[0]
    if neg_area_idx.size:
        warnings.warn("Negative areas founds: %d" % neg_area_idx.size, RuntimeWarning)
    #TODO proper warm up and learning rate may reduce the prob of assertion fail
    # assert np.all(areas >= 0), 'Negative areas founds'
    return areas, neg_area_idx


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def filter_small_boxes(boxes, min_size):
    """Keep boxes with width and height both greater than min_size."""
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((w > min_size) & (h > min_size))[0]
    return keep


def clip_boxes_to_image(boxes, height, width):
    """Clip an array of boxes to an image with the given height and width."""
    boxes[:, [0, 2]] = np.minimum(width - 1., np.maximum(0., boxes[:, [0, 2]]))
    boxes[:, [1, 3]] = np.minimum(height - 1., np.maximum(0., boxes[:, [1, 3]]))
    return boxes


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 6 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 6.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::6] = np.maximum(np.minimum(boxes[:, 0::6], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::6] = np.maximum(np.minimum(boxes[:, 1::6], im_shape[0] - 1), 0)
    # z1 >= 0
    boxes[:, 2::6] = np.maximum(np.minimum(boxes[:, 2::6], im_shape[2] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 3::6] = np.maximum(np.minimum(boxes[:, 3::6], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 4::6] = np.maximum(np.minimum(boxes[:, 4::6], im_shape[0] - 1), 0)
    # z2 < im_shape[2]
    boxes[:, 5::6] = np.maximum(np.minimum(boxes[:, 5::6], im_shape[2] - 1), 0)

    return boxes

# def clip_tiled_boxes(boxes, window):
#     """
#     boxes: [N, 6] each col is y1, x1, y2, x2, z1, z2
#     window: [6] in the form y1, x1, y2, x2, z1, z2
#     """
#     boxes = np.concatenate(
#         (np.clip(boxes[:, 0], 0, window[0])[:, None],
#          np.clip(boxes[:, 3], 0, window[0])[:, None],
#          np.clip(boxes[:, 1], 0, window[1])[:, None],
#          np.clip(boxes[:, 4], 0, window[1])[:, None],
#          np.clip(boxes[:, 2], 0, window[2])[:, None],
#          np.clip(boxes[:, 5], 0, window[2])[:, None]), 1
#     )
#     return boxes


# def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
#     """Applies the given deltas to the given boxes.
#     boxes: [N, 6] where each row is y1, x1, y2, x2, z1, z2
#     deltas: [N, 6] where each row is [dy, dx, dz, log(dh), log(dw), log(dd)]
#     """
#     # Convert to y, x, h, w
#     boxes = torch.from_numpy(boxes)
#     deltas = torch.from_numpy(deltas)
#     height = boxes[:, 4] - boxes[:, 1]
#     width = boxes[:, 3] - boxes[:, 0]
#     depth = boxes[:, 5] - boxes[:, 2]
#     center_y = boxes[:, 1] + 0.5 * height
#     center_x = boxes[:, 0] + 0.5 * width
#     center_z = boxes[:, 2] + 0.5 * depth
#     # Apply deltas
#     center_y += deltas[:, 1] * height
#     center_x += deltas[:, 0] * width
#     center_z += deltas[:, 2] * depth
#     height *= torch.exp(deltas[:, 4])
#     width *= torch.exp(deltas[:, 3])
#     depth *= torch.exp(deltas[:, 5])
#     # Convert back to y1, x1, y2, x2
#     y1 = center_y - 0.5 * height
#     x1 = center_x - 0.5 * width
#     z1 = center_z - 0.5 * depth
#     y2 = y1 + height
#     x2 = x1 + width
#     z2 = z1 + depth
#     result = torch.stack([x1, y1, z1, x2, y2, z2], dim=1)
#     result = result.cpu().numpy()
#     return result


def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 3] - boxes[:, 0] + 1.0
    heights = boxes[:, 4] - boxes[:, 1] + 1.0
    depths = boxes[:, 5] - boxes[:, 2] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    ctr_z = boxes[:, 2] + 0.5 * depths

    wx, wy, wz, ww, wh, wd = weights
    dx = deltas[:, 0::6] / wx
    dy = deltas[:, 1::6] / wy
    dz = deltas[:, 2::6] / wz
    dw = deltas[:, 3::6] / ww
    dh = deltas[:, 4::6] / wh
    dd = deltas[:, 5::6] / wd

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, np.log(1000. / 16.))
    dh = np.minimum(dh, np.log(1000. / 16.))
    dd = np.minimum(dd, np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_ctr_z = dz * depths[:, np.newaxis] + ctr_z[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_d = np.exp(dd) * depths[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::6] = pred_ctr_y - 0.5 * pred_h
    # z1
    pred_boxes[:, 2::6] = pred_ctr_z - 0.5 * pred_d
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::6] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 4::6] = pred_ctr_y + 0.5 * pred_h - 1
    # z2
    pred_boxes[:, 5::6] = pred_ctr_z + 0.5 * pred_d - 1

    return pred_boxes


def bbox_transform_inv(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 3] - boxes[:, 0] + 1.
    ex_heights = boxes[:, 4] - boxes[:, 1] + 1.
    ex_depths = boxes[:, 5] - boxes[:, 2] + 1.
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights
    ex_ctr_z = boxes[:, 2] + 0.5 * ex_depths

    gt_widths = gt_boxes[:, 3] - gt_boxes[:, 0] + 1.
    gt_heights = gt_boxes[:, 4] - gt_boxes[:, 1] + 1.
    gt_depths = gt_boxes[:, 5] - gt_boxes[:, 2] + 1.
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights
    gt_ctr_z = gt_boxes[:, 2] + 0.5 * gt_depths

    wx, wy, wz, ww, wh, wd = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dz = wz * (gt_ctr_z - ex_ctr_z) / ex_depths
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)
    targets_dd = wd * np.log(gt_depths / ex_depths)

    targets = np.vstack((targets_dx, targets_dy, targets_dz, targets_dw,
                         targets_dh, targets_dd)).transpose()
    return targets

# def bbox_transform_inv(box, gt_box, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
#     """Compute refinement needed to transform box to gt_box.
#     box and gt_box are [N, (y1, x1, y2, x2)] / 3D: (z1, z2))
#     """
#     box = torch.from_numpy(box)
#     gt_box = torch.from_numpy(gt_box)
#     height = box[:, 4] - box[:, 1] + 1
#     width = box[:, 3] - box[:, 0] + 1
#     center_y = box[:, 1] + 0.5 * height
#     center_x = box[:, 0] + 0.5 * width

#     gt_height = gt_box[:, 4] - gt_box[:, 1] + 1
#     gt_width = gt_box[:, 3] - gt_box[:, 0] + 1
#     gt_center_y = gt_box[:, 1] + 0.5 * gt_height
#     gt_center_x = gt_box[:, 0] + 0.5 * gt_width

#     dy = (gt_center_y - center_y) / height
#     dx = (gt_center_x - center_x) / width
#     dh = torch.log(gt_height / height)
#     dw = torch.log(gt_width / width)

#     depth = box[:, 5] - box[:, 2] + 1
#     center_z = box[:, 2] + 0.5 * depth
#     gt_depth = gt_box[:, 5] - gt_box[:, 2] + 1
#     gt_center_z = gt_box[:, 2] + 0.5 * gt_depth
#     dz = (gt_center_z - center_z) / depth
#     dd = torch.log(gt_depth / depth)
#     result = torch.stack([dx, dy, dz, dw, dh, dd], dim=1)

#     result = result.cpu().numpy()
#     return result


def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 3] - boxes[:, 0]) * .5
    h_half = (boxes[:, 4] - boxes[:, 1]) * .5
    d_half = (boxes[:, 5] - boxes[:, 2]) * .5
    x_c = (boxes[:, 3] + boxes[:, 0]) * .5
    y_c = (boxes[:, 4] + boxes[:, 1]) * .5
    z_c = (boxes[:, 5] + boxes[:, 2]) * .5

    w_half *= scale
    h_half *= scale
    d_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 3] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 4] = y_c + h_half
    boxes_exp[:, 2] = z_c - d_half
    boxes_exp[:, 5] = z_c + d_half

    return boxes_exp


def flip_boxes(boxes, im_width):
    """Flip boxes horizontally."""
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped

def aspect_ratio(boxes, aspect_ratio):
    """Perform width-relative aspect ratio transformation."""
    boxes_ar = boxes.copy()
    boxes_ar[:, 0::4] = aspect_ratio * boxes[:, 0::4]
    boxes_ar[:, 2::4] = aspect_ratio * boxes[:, 2::4]
    return boxes_ar


def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out

def nms(boxes, scores, iou_threshold):
    scores = scores.reshape((scores.shape[0]))
    return c_nms.nms(torch.from_numpy(boxes).cuda(), torch.from_numpy(scores).cuda(), iou_threshold)

# def nms(dets, thresh):
#     """Apply classic DPM-style greedy NMS."""
#     if dets.shape[0] == 0:
#         return []
#     return cython_nms.nms(dets, thresh)

# def nms(box_coords, scores, thresh):
#     """ non-maximum suppression on 2D or 3D boxes in numpy.
#     :param box_coords: [y1,x1,y2,x2 (,z1,z2)] with y1<=y2, x1<=x2, z1<=z2.
#     :param scores: ranking scores (higher score == higher rank) of boxes.
#     :param thresh: IoU threshold for clustering.
#     :return:
#     """
#     y1 = box_coords[:, 1]
#     x1 = box_coords[:, 0]
#     y2 = box_coords[:, 4]
#     x2 = box_coords[:, 3]
#     assert np.all(y1 <= y2) and np.all(x1 <= x2), """"the definition of the coordinates is crucially important here:
#             coordinates of which maxima are taken need to be the lower coordinates"""
#     areas = (x2 - x1) * (y2 - y1)

#     is_3d = box_coords.shape[1] == 6
#     if is_3d: # 3-dim case
#         z1 = box_coords[:, 2]
#         z2 = box_coords[:, 5]
#         assert np.all(z1<=z2), """"the definition of the coordinates is crucially important here:
#            coordinates of which maxima are taken need to be the lower coordinates"""
#         areas *= (z2 - z1)

#     order = scores.argsort(axis=0)[::-1]

#     keep = []
#     while order.size > 0:  # order is the sorted index.  maps order to index: order[1] = 24 means (rank1, ix 24)
#         i = order[0] # highest scoring element
#         yy1 = np.maximum(y1[i], y1[order])  # highest scoring element still in >order<, is compared to itself, that is okay.
#         xx1 = np.maximum(x1[i], x1[order])
#         yy2 = np.minimum(y2[i], y2[order])
#         xx2 = np.minimum(x2[i], x2[order])

#         h = np.maximum(0.0, yy2 - yy1)
#         w = np.maximum(0.0, xx2 - xx1)
#         inter = h * w

#         if is_3d:
#             zz1 = np.maximum(z1[i], z1[order])
#             zz2 = np.minimum(z2[i], z2[order])
#             d = np.maximum(0.0, zz2 - zz1)
#             inter *= d

#         iou = inter / (areas[i] + areas[order] - inter)

#         non_matches = np.nonzero(iou <= thresh)[0]  # get all elements that were not matched and discard all others.
#         order = order[non_matches]
#         keep.append(i)

#     return keep


def soft_nms(
    dets, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear'
):
    """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
    if dets.shape[0] == 0:
        return dets, []

    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)

    dets, keep = cython_nms.soft_nms(
        np.ascontiguousarray(dets, dtype=np.float32),
        np.float32(sigma),
        np.float32(overlap_thresh),
        np.float32(score_thresh),
        np.uint8(methods[method])
    )
    return dets, keep
