import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import sparseconvnet as scn
import mlreco.models.utils.boxes as box_utils
import mlreco.models.utils.net as net_utils
import mlreco.models.utils.blob as blob_utils
import mlreco.models.fast_rcnn as fast_rcnn
from mlreco.models.nms import nms
import plotly.graph_objs as go
import plotly
import scipy
import time

import threading
from collections import namedtuple

# Cache for memoizing _get_field_of_anchors
_threadlocal_foa = threading.local()

FieldOfAnchors = namedtuple(
    'FieldOfAnchors', [
        'field_of_anchors', 'num_cell_anchors', 'stride', 'field_size',
        'octave', 'aspect'
    ]
)

class RPN(nn.Module):

    def __init__(self, cfg, name="rpn"):
        super(RPN,self).__init__()
        self.model_config = cfg[name]
        self.validation = False
        self.dim_in = self.model_config.get('dim_in', 1024)
        self.dim_out = self.model_config.get('dim_out', 1024)
        self.spatial_scale = 1/16
        anchors = generate_anchors(
            stride=1. / self.spatial_scale,
            scales_xy=((32,64,128,256,512)),
            scales_z=(32,64,128,256,512),
            aspect_ratios=(0.5,1,2))
#         anchors = generate_anchors_3D((32, 64, 128, 256, 512), (32, 64, 128, 256, 512), (0.5,1,2), (65, 65, 65), 16, 16, 16)
        num_anchors = anchors.shape[0]

        # RPN hidden representation
#         self.RPN_conv = nn.Conv3d(self.dim_in, self.dim_out, 3, 1, 1)
        self.RPN_conv = scn.SubmanifoldConvolution(3, self.dim_in, self.dim_out, 3, True)
        # Proposal classification scores
        self.n_score_out = num_anchors
#         self.RPN_cls_score = nn.Conv3d(self.dim_out, self.n_score_out, 1, 1, 0)
        self.RPN_cls_score = scn.SubmanifoldConvolution(3, self.dim_out, self.n_score_out, 1, True)
        # Proposal bbox regression deltas
#         self.RPN_bbox_pred = nn.Conv3d(self.dim_out, num_anchors * 6, 1, 1, 0)
        self.RPN_bbox_pred = scn.SubmanifoldConvolution(3, self.dim_out, num_anchors * 6, 1, True)
    
        self.densify_conv = scn.SparseToDense(3, self.dim_out)
        self.densify_cls = scn.SparseToDense(3, self.n_score_out)
        self.densify_bbox = scn.SparseToDense(3, num_anchors * 6)

        self.RPN_GenerateProposals = GenerateProposalsOp(self.model_config, anchors, self.spatial_scale, self.validation)
        self.RPN_GenerateProposalLabels = GenerateProposalLabelsOp()
        self.ReLU = scn.ReLU()

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.RPN_conv.weight, std=0.01)
#         init.constant_(self.RPN_conv.bias, 0)
        init.normal_(self.RPN_cls_score.weight, std=0.01)
#         init.constant_(self.RPN_cls_score.bias, 0)
        init.normal_(self.RPN_bbox_pred.weight, std=0.01)
#         init.constant_(self.RPN_bbox_pred.bias, 0)


    def forward(self, x, im_info, roidb=None, input_data=None, gt_boxes=None):
        """
        x: feature maps from the backbone network. (Variable)
        im_info: (CPU Variable)
        roidb: (list of ndarray)
        """

        rpn_conv = self.ReLU(self.RPN_conv(x))
        
        rpn_cls_logits = self.RPN_cls_score(rpn_conv)

        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv)

        rpn_cls_logits = self.densify_cls(rpn_cls_logits)
        rpn_bbox_pred = self.densify_bbox(rpn_bbox_pred)
        
        return_dict = {
            'rpn_cls_logits': rpn_cls_logits, 'rpn_bbox_pred': rpn_bbox_pred}

        # Proposals are needed during:
        #  1) inference (== not model.train) for RPN only and Faster R-CNN
        #  OR
        #  2) training for Faster R-CNN
        # Otherwise (== training for RPN only), proposals are not needed
        # rpn_cls_prob = F.sigmoid(rpn_cls_logits)
        rpn_cls_prob = torch.sigmoid(rpn_cls_logits)
        rpn_rois, rpn_rois_prob = self.RPN_GenerateProposals(
        rpn_cls_prob, rpn_bbox_pred, im_info, input_data, gt_boxes)

        return_dict['rpn_rois'] = rpn_rois
        return_dict['rpn_roi_probs'] = rpn_rois_prob

        if self.training or self.validation:
#             Add op that generates training labels for in-network RPN proposals
            blobs_out = self.RPN_GenerateProposalLabels(rpn_rois, roidb, im_info)
            return_dict.update(blobs_out)
        if not self.training:
            # Alias rois to rpn_rois for inference
            return_dict['rois'] = return_dict['rpn_rois']
        return return_dict


def generate_anchors(
    stride=16, scales_xy=(32, 64, 128, 256, 512), scales_z=(8, 16, 32, 64, 128), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(scales_xy, dtype=np.float) / stride,
        np.array(scales_z, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float)
    )


def _generate_anchors(base_size, scales_xy, scales_z, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, 1, base_size, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, scales_xy, scales_z, aspect_ratios)
#     anchors = np.vstack(
#         [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
#     )
    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[3] - anchor[0] + 1
    h = anchor[4] - anchor[1] + 1
    d = anchor[5] - anchor[2] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    z_ctr = anchor[2] + 0.5 * (d - 1)
    return w, h, d, x_ctr, y_ctr, z_ctr


def _mkanchors(ws, hs, ds, x_ctr, y_ctr, z_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    ds = ds[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            z_ctr - 0.5 * (ds - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
            z_ctr + 0.5 * (ds - 1)
        )
    )
    return anchors


def _ratio_enum(anchor, scales_xy, scales_z, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    # w, h, x_ctr, y_ctr = _whctrs(anchor)
    w, h, d, x_ctr, y_ctr, z_ctr = _whctrs(anchor)
    size = w * h * d
#     size_xy = w * h
#     size_ratios_xy = size_xy / ratios
    
#     size_ratios_z = d / [2, 1, 0.5]
            
    scales_xy, ratios_meshed = np.meshgrid(np.array(scales_xy), np.array(ratios))
#     scales_xy, scales_z, ratios_meshed = np.meshgrid(np.array(scales_xy), np.array(scales_z), np.array(ratios))

    scales_xy = scales_xy.flatten()
    ratios_meshed = ratios_meshed.flatten()

#     Enumerate heights and widths from scales and ratios
    hs = (h * (scales_xy / np.sqrt(ratios_meshed)))
    ws = (w * (scales_xy * np.sqrt(ratios_meshed)))
    ds = (d * (np.tile(np.array(scales_z), len(ratios_meshed) // np.array(scales_z)[..., None].shape[0])))

#     ds = d * (scales_z * np.sqrt(ratios_meshed))

#     ws = np.round(np.cbrt(size_ratios))
#     hs = np.round(ws * (ratios[:,0]))
#     ds = np.round(ws * (ratios[:,1]))
#     anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    anchors = _mkanchors(ws, hs, ds, x_ctr, y_ctr, z_ctr)
    return anchors


class GenerateProposalsOp(nn.Module):
    def __init__(self, cfg, anchors, spatial_scale, validation=False):
        super(GenerateProposalsOp,self).__init__()
        self.model_config = cfg
        self._anchors = anchors
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = 1. / spatial_scale
        self.validation = validation

    def forward(self, rpn_cls_prob, rpn_bbox_pred, im_info, input_data, gt_boxes):
        """Op for generating RPN porposals.
        blobs_in:
          - 'rpn_cls_probs': 4D tensor of shape (N, A, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].
          - 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.
          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.
        blobs_out:
          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        # 1. for each location i in a (H, W) grid:
        #      generate A anchor boxes centered on cell i
        #      apply predicted bbox deltas to each of the A anchors at cell i
        # 2. clip predicted boxes to image
        # 3. remove predicted boxes with either height or width < threshold
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take the top pre_nms_topN proposals before NMS
        # 6. apply NMS with a loose threshold (0.7) to the remaining proposals
        # 7. take after_nms_topN proposals after NMS
        # 8. return the top proposals

        """Type conversion"""
        # predicted probability of fg object for each RPN anchor

        scores = rpn_cls_prob.data.cpu().numpy()
        # predicted achors transformations
        bbox_deltas = rpn_bbox_pred.data.cpu().numpy()
        # input image (height, width, scale), in which scale is the scale factor
        # applied to the original dataset image to get the network input image
        im_info = im_info.data.cpu().numpy()

        # 1. Generate proposals from bbox deltas and shifted anchors
#         height, width, depth = scores.shape[-3:]
        # Enumerate all shifted positions on the (H, W) grid
#         coords = input_data[:, :3+1].float().cpu().numpy()
#         vals = input_data[:, 3+1:].float().cpu().numpy()
#         over_thresh = np.where(vals > 1.)[0]
#         coords = coords[over_thresh, :]
#         coords = (coords/16.)[:, :3]

#         coords = rpn_cls_prob.nonzero()[:, 2:].cpu().numpy()
#         coords = rpn_bbox_pred.mean(dim=1).nonzero().cpu().numpy()[:, 1:]
        
#         coords = conv_feat.cpu().numpy()
        
#         shift_x = np.unique(np.round(coords)[:,0]) * self._feat_stride
#         shift_y = np.unique(np.round(coords)[:,1]) * self._feat_stride
#         shift_z = np.unique(np.round(coords)[:,2]) * self._feat_stride

        shift_x = np.arange(0, 64) * self._feat_stride
        shift_y = np.arange(0, 64) * self._feat_stride
        shift_z = np.arange(0, 64) * self._feat_stride
        
        shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z, copy=False)
        # Convert to (K, 4), K=H*W, where the columns are (dx, dy, dx, dy)
        # shift pointing to each grid location
#         shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_z.ravel(), shift_x.ravel(),
#                             shift_y.ravel(), shift_z.ravel())).transpose()
#         scores = rpn_cls_prob.data.cpu().numpy().take(np.unique(np.round(coords)[:,0]).astype(np.int8), axis=2).take(np.unique(np.round(coords)[:,1]).astype(np.int8), axis=3).take(np.unique(np.round(coords)[:,2]).astype(np.int8),axis=4)
#         bbox_deltas = rpn_bbox_pred.data.cpu().numpy().take(np.unique(np.round(coords)[:,0]).astype(np.int8), axis=2).take(np.unique(np.round(coords)[:,1]).astype(np.int8), axis=3).take(np.unique(np.round(coords)[:,2]).astype(np.int8),axis=4)
        
#         coords = (coords * 16).astype(np.int32)
        
#         widths = self._anchors[:,3] - self._anchors[:,0]
#         heights = self._anchors[:,4] - self._anchors[:,1]
#         depths = self._anchors[:,5] - self._anchors[:,2]
#         box_sizes = np.stack([widths, heights, depths]).reshape([-1, 3])
        
#         all_anchors = []
#         scores_l = []
#         bbox_deltas_l = []
        
#         for i in coords:
#             scores_l.append(scores[:, :, i[0] // 16, i[1] // 16, i[2] // 16])
#             bbox_deltas_l.append(bbox_deltas[:, :, i[0] // 16, i[1] // 16, i[2] // 16])
#             all_anchors.append(np.hstack((i - 0.5 * box_sizes, i + 0.5 * box_sizes)))
            
#         all_anchors = np.array(all_anchors).reshape(-1, 6)
#         scores = np.array(scores_l).reshape(-1, 1)
#         bbox_deltas = np.array(bbox_deltas_l).reshape(-1, 6)
    
#         scores = scores[:, :, np.unique(np.round(coords)[:,0]).astype(np.int8), :, :]
#         scores = scores[:, :, :, np.unique(np.round(coords)[:,0]).astype(np.int8), :]
#         scores = scores[:, :, :, :, np.unique(np.round(coords)[:,0]).astype(np.int8)]

#         bbox_deltas = bbox_deltas[:, :, np.unique(np.round(coords)[:,0]).astype(np.int8), :, :]
#         bbox_deltas = bbox_deltas[:, :, :, np.unique(np.round(coords)[:,0]).astype(np.int8), :]
#         bbox_deltas = bbox_deltas[:, :, :, :, np.unique(np.round(coords)[:,0]).astype(np.int8)]
        # Broacast anchors over shifts to enumerate all anchors at all positions
        # in the (H, W) grid:
        #   - add A anchors of shape (1, A, 4) to
        #   - K shifts of shape (K, 1, 4) to get
        #   - all shifted anchors of shape (K, A, 4)
        #   - reshape to (K*A, 4) shifted anchors
        num_images = 1
#         A = self._num_anchors
#         K = shifts.shape[0]
#         all_anchors = self._anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
#         all_anchors = all_anchors.reshape((K * A, 6))
#         all_anchors = torch.from_numpy(all_anchors).type_as(scores)
        
        box_widths, box_centers_x = np.meshgrid(self._anchors[:,3] - self._anchors[:,0], shift_x)
        box_heights, box_centers_y = np.meshgrid(self._anchors[:,4] - self._anchors[:,1], shift_y)
        box_depths, box_centers_z = np.meshgrid(self._anchors[:,5] - self._anchors[:,2], shift_z)
        
        # Reshape to get a list of (y, x, z) and a list of (h, w, d)
        box_centers = np.stack(
            [box_centers_y, box_centers_x, box_centers_z], axis=2).reshape([-1, 3])
        box_sizes = np.stack([box_heights, box_widths, box_depths], axis=2).reshape([-1, 3])
        
        # Convert to corner coordinates (y1, x1, y2, x2, z1, z2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        
        boxes = np.transpose(np.array([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 2], boxes[:, 5]]), axes=(1, 0))
        order = np.array([1, 0, 4, 3, 2, 5])
        all_anchors = boxes[:, order]
        
#         coords = coords.astype(np.int32)
        
#         t = np.zeros((15, 64, 64, 64))
#         for i in np.unique(coords[:,0]):
#             for j in np.unique(coords[:,1]):
#                 for k in np.unique(coords[:,2]):
#                     t[:, i, j, k] = 1.
#         inds_inside = t.reshape(-1).nonzero()[0]
#         all_anchors = all_anchors[inds_inside, :]

#         coords = coords.astype(np.int32)
#         all_anchors_l = []
#         scores_l = []
#         bbox_deltas_l = []
#         all_anchors = all_anchors.reshape(-1, 64, 64, 64, 6)
#         for i in coords:
#             all_anchors_l.append(all_anchors[:, i[0], i[1], i[2], :])
#             scores_l.append(scores[:, :, i[0], i[1], i[2]])
#             bbox_deltas_l.append(bbox_deltas[:, :, i[0], i[1], i[2]])

#         all_anchors = np.array(all_anchors_l).reshape(-1,6)
#         scores = np.array(scores_l).reshape(-1, 1)
#         bbox_deltas = np.array(bbox_deltas_l).reshape(-1, 6)
        
        print(all_anchors.shape)
        rois = np.empty((0, 7), dtype=np.float32)
        roi_probs = np.empty((0, 1), dtype=np.float32)
        for im_i in range(num_images):
            im_i_boxes, im_i_probs = self.proposals_for_one_image(
                im_info[im_i, :], all_anchors, bbox_deltas[im_i, :, :, :, :],
                scores[im_i, :, :, :, :], input_data, gt_boxes)
            batch_inds = im_i * np.ones(
                (im_i_boxes.shape[0], 1), dtype=np.float32)
            im_i_rois = np.hstack((batch_inds, im_i_boxes))
            rois = np.append(rois, im_i_rois, axis=0)
            roi_probs = np.append(roi_probs, im_i_probs, axis=0)

        return rois, roi_probs  # Note: ndarrays

    def proposals_for_one_image(self, im_info, all_anchors, bbox_deltas, scores, input_data, gt_boxes):
        # Get mode-dependent configuration
#         cfg_key = 'TRAIN' if self.training else 'TEST'
#         pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
#         post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
#         nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
#         min_size = cfg[cfg_key].RPN_MIN_SIZE
        pre_nms_topN = self.model_config.get('pre_nms_topN', 192000)
        post_nms_topN = self.model_config.get('post_nms_topN', 32000)
        nms_thresh = self.model_config.get('nms_thresh', 0.7)
        min_size = 1
        # print('generate_proposals:', pre_nms_topN, post_nms_topN, nms_thresh, min_size)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #   - bbox deltas will be (4 * A, H, W) format from conv output
        #   - transpose to (H, W, 4 * A)
        #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
        #     in slowest to fastest order to match the enumerated anchors
        scores = scores.transpose((1, 2, 3, 0)).reshape((-1, 1))
        active_inds = np.where(scores != 0.5)[0]
        scores = scores[active_inds]
        
        bbox_deltas = bbox_deltas.transpose((1, 2, 3, 0)).reshape((-1, 6))[active_inds, :]
        all_anchors = all_anchors[active_inds, :]
        
#         bbox_deltas = bbox_deltas * np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        # Same story for the scores:
        #   - scores are (A, H, W) format from conv output
        #   - transpose to (H, W, A)
        #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
        #     to match the order of anchors and bbox_deltas
        
#         overlaps = box_utils.bbox_overlaps(all_anchors.astype(dtype=np.float32), gt_boxes.astype(dtype=np.float32))
#         anchor_to_gt_argmax = overlaps.argmax(axis=1)
#         anchor_to_gt_max = overlaps[np.arange(all_anchors.shape[0]),
#                                                 anchor_to_gt_argmax]
        
#         gt_to_anchor_argmax = overlaps.argmax(axis=0)
#         # For each gt box, amount of overlap with most overlapping anchor
#         gt_to_anchor_max = overlaps[
#             gt_to_anchor_argmax,
#             np.arange(overlaps.shape[1])
#         ]
#         # Find all anchors that share the max overlap amount
#         # (this includes many ties)
#         anchors_with_max_overlap = np.where(
#             overlaps == gt_to_anchor_max
#         )[0]
        
#         print(gt_to_anchor_max)
        
#         inds = np.hstack((anchors_with_max_overlap, np.where(anchor_to_gt_max >= 0.25)[0]))
        
#         all_anchors = all_anchors[inds, :]
#         bbox_deltas = bbox_deltas[inds, :]
#         scores = scores[inds, :]
        
        # print('pre_nms:', bbox_deltas.shape, scores.shape)
        
#         proposals = box_utils.bbox_transform(all_anchors[fg_inds], bbox_deltas[fg_inds])
#         proposals = box_utils.clip_tiled_boxes(proposals, im_info[:3])
        
#         return proposals, scores[fg_inds]
        
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)

        if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
            order = np.argsort(-scores.squeeze())
        else:
            # Avoid sorting possibly large arrays; First partition to get top K
            # unsorted and then sort just those (~20x faster for 200k scores)
            inds = np.argpartition(-scores.squeeze(),
                                   pre_nms_topN)[:pre_nms_topN]
            order = np.argsort(-scores[inds].squeeze())
            order = inds[order]
        bbox_deltas = bbox_deltas[order, :]
        all_anchors = all_anchors[order, :]
        scores = scores[order]

        # Transform anchors into proposals via bbox transformations

        proposals = box_utils.bbox_transform(all_anchors, bbox_deltas, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        print(bbox_deltas.max())
        print(bbox_deltas.min())
        print(bbox_deltas.std())

        # 2. clip proposals to image (may result in proposals with zero area
        # that will be removed in the next step)
        proposals = box_utils.clip_tiled_boxes(proposals, im_info[:3])
        # 3. remove predicted boxes with either height or width < min_size
        
        keep = _filter_boxes(proposals, min_size, im_info)
        proposals = proposals[keep, :]
        scores = scores[keep]
        # print('pre_nms:', proposals.shape, scores.shape)

        # 6. apply loose nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if nms_thresh > 0:
#             keep = nms.nms(torch.from_numpy(proposals).cuda(), torch.from_numpy(scores).cuda(), nms_thresh)
            keep = box_utils.nms(proposals, scores, nms_thresh)
            keep = keep.cpu().numpy()
            keep = np.array(keep).flatten().astype(np.int32)
            print("NUM PROPOSALS: ", keep.shape)

            # print('nms keep:', keep.shape)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]
   
            proposals = proposals[keep, :]
            scores = scores[keep]

        rpn_vis = self.model_config.get('rpn_vis', False)
        
        if rpn_vis:
            keep = np.where(np.logical_and.reduce((proposals[:, 0] > 0, proposals[:, 3] < 350, proposals[:, 1] > 550, proposals[:, 4] < 875, proposals[:, 2] > 450, proposals[:, 5] < 900)))[0]
            above99 = np.where(scores > 0.99)[0]
            boxes = proposals[:30, :]
            scores_v = scores[:30, :]
            
            import pdb; pdb.set_trace()
            
            gt_class = np.zeros((boxes.shape[0], 1))
            
            scores_v = np.ones((boxes.shape[0], 1))
            cls_boxes = np.hstack((boxes, scores_v, gt_class))
            
            vis_data([], cls_boxes, input_data, [])
            import pdb; pdb.set_trace()
        return proposals, scores


def _filter_boxes(boxes, min_size, im_info):
    """Only keep boxes with both sides >= min_size and center within the image.
  """
    # Scale min_size to match image scale
    min_size *= im_info[3]
    ws = boxes[:, 3] - boxes[:, 0]
    hs = boxes[:, 4] - boxes[:, 1]
    ds = boxes[:, 5] - boxes[:, 2]
    x_ctr = boxes[:, 0] + ws / 2.
    y_ctr = boxes[:, 1] + hs / 2.
    z_ctr = boxes[:, 2] + ds / 2.
    keep = np.where((ws > min_size) & (hs > min_size) & (ds > min_size) &
                    (x_ctr < im_info[1]) & (y_ctr < im_info[0]) & (z_ctr < im_info[2]))[0]
    return keep


class GenerateProposalLabelsOp(nn.Module):
    def __init__(self):
        super(GenerateProposalLabelsOp,self).__init__()

    def forward(self, rpn_rois, roidb, im_info):
        """Op for generating training labels for RPN proposals. This is used
        when training RPN jointly with Fast/Mask R-CNN (as in end-to-end
        Faster R-CNN training).
        blobs_in:
          - 'rpn_rois': 2D tensor of RPN proposals output by GenerateProposals
          - 'roidb': roidb entries that will be labeled
          - 'im_info': See GenerateProposals doc.
        blobs_out:
          - (variable set of blobs): returns whatever blobs are required for
            training the model. It does this by querying the data loader for
            the list of blobs that are needed.
        """
        im_scales = im_info.data.numpy()[:, 3]

        output_blob_names = fast_rcnn.get_fast_rcnn_blob_names()
        # For historical consistency with the original Faster R-CNN
        # implementation we are *not* filtering crowd proposals.
        # This choice should be investigated in the future (it likely does
        # not matter).
        # Note: crowd_thresh=0 will ignore _filter_crowd_proposals
        # coco:
        # json_dataset.add_proposals(roidb, rpn_rois, im_scales, crowd_thresh=0)
        # root:

        add_proposals(roidb, rpn_rois, im_scales, crowd_thresh=0)

        # json_dataset.add_proposals(roidb, rpn_rois, im_scales, crowd_thresh=0)
        blobs = {k: [] for k in output_blob_names}
        fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)

        return blobs


# In[49]:


def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """

    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(np.float32),
                gt_boxes.astype(np.float32)
            )
#             proposal_to_gt_overlaps = (box_utils.overlaps_3D(
#                 torch.from_numpy(boxes).float(),
#                 torch.from_numpy(gt_boxes).float()
#             )).numpy()
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        # print("entry['boxes'].shape",entry['boxes'].shape)
        # print("boxes.astype(entry['boxes'].dtype, copy=False).shape",boxes.astype(entry['boxes'].dtype, copy=False).shape)

        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
#   gt
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )
def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)
        
def get_rpn_blob_names(is_training=True):
    """Blob names used by RPN."""
    # im_info: (height, width, image scale)
    blob_names = ['im_info']
    if is_training:
        # gt boxes: (batch_idx, x1, y1, x2, y2, cls)
        blob_names += ['roidb']
            # Single level RPN blobs
        blob_names += [
            'rpn_labels_int32_wide',
            'rpn_bbox_targets_wide',
            'rpn_bbox_inside_weights_wide',
            'rpn_bbox_outside_weights_wide',
            'anchors'
        ]
    return blob_names

def add_rpn_blobs(blobs, im_scales, roidb, input_data):
    """Add blobs needed training RPN-only and end-to-end Faster R-CNN models."""

#     foa = data_utils.get_field_of_anchors(cfg.RPN.STRIDE, cfg.RPN.SIZES,
#                                           cfg.RPN.ASPECT_RATIOS)
    foa = get_field_of_anchors(16, (32,64,128,256,512), (32,64,128,256,512), (0.5,1,2))

    all_anchors = foa.field_of_anchors


    for im_i, entry in enumerate(roidb):
        scale = im_scales[im_i]
        im_height = np.round(1024 * scale)
        im_width = np.round(1024 * scale)
        im_depth = np.round(1024 * scale)
        gt_inds = np.where(
            (entry['gt_classes'] > 0)
        )[0]
        
        gt_rois = entry['boxes'][gt_inds, :] * scale
        # TODO(rbg): gt_boxes is poorly named;
        # should be something like 'gt_rois_info'
        gt_boxes = blob_utils.zeros((len(gt_inds), 8))
        gt_boxes[:, 0] = im_i  # batch inds
        gt_boxes[:, 1:7] = gt_rois
        gt_boxes[:, 7] = entry['gt_classes'][gt_inds]
        im_info = np.array([[im_height, im_width, im_depth, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)

        # Add RPN targets
        # Classical RPN, applied to a single feature level
        rpn_blobs = _get_rpn_blobs(
            im_height, im_width, im_depth, [foa], all_anchors, gt_rois, input_data
        )
        for k, v in rpn_blobs.items():
            blobs[k].append(v)

    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

    valid_keys = [
        'has_visible_keypoints', 'boxes', 'segms', 'seg_areas', 'gt_classes',
        'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map', 'gt_keypoints', 'chain_cluster',
        'id', 'plane'
    ]
    minimal_roidb = [{} for _ in range(len(roidb))]
    for i, e in enumerate(roidb):
        for k in valid_keys:
            if k in e:
                minimal_roidb[i][k] = e[k]
    # blobs['roidb'] = blob_utils.serialize(minimal_roidb)
    blobs['roidb'] = minimal_roidb

    # Always return valid=True, since RPN minibatches are valid by design
    return True

def _get_rpn_blobs(im_height, im_width, im_depth, foas, all_anchors, gt_boxes, input_data):
    
    coords = input_data[:, :3+1].float().cpu().numpy()
    vals = input_data[:, 3+1:].float().cpu().numpy()
    over_thresh = np.where(vals > 1.)[0]
    coords = coords[over_thresh, :]
    coords = (coords/16.)[:, :3]
    
    cell_anchors = generate_anchors(
            stride=16.,
            scales_xy=((32,64,128,256,512)),
            scales_z=(32,64,128,256,512),
            aspect_ratios=(0.5,1,2))
    
    orig_anchors = all_anchors
    total_anchors = all_anchors.shape[0]
    
    shift_x = np.unique(np.round(coords)[:,0]) * 16.
    shift_y = np.unique(np.round(coords)[:,1]) * 16.
    shift_z = np.unique(np.round(coords)[:,2]) * 16.

#     shift_x = np.arange(0, 64) * 16.
#     shift_y = np.arange(0, 64) * 16.
#     shift_z = np.arange(0, 64) * 16.

    shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z, copy=False)
    # Convert to (K, 4), K=H*W, where the columns are (dx, dy, dx, dy)
    # shift pointing to each grid location

    # Broacast anchors over shifts to enumerate all anchors at all positions
    # in the (H, W) grid:
    #   - add A anchors of shape (1, A, 4) to
    #   - K shifts of shape (K, 1, 4) to get
    #   - all shifted anchors of shape (K, A, 4)
    #   - reshape to (K*A, 4) shifted anchors
    num_images = 1

    box_widths, box_centers_x = np.meshgrid(cell_anchors[:,3] - cell_anchors[:,0], shift_x)
    box_heights, box_centers_y = np.meshgrid(cell_anchors[:,4] - cell_anchors[:,1], shift_y)
    box_depths, box_centers_z = np.meshgrid(cell_anchors[:,5] - cell_anchors[:,2], shift_z)

    # Reshape to get a list of (y, x, z) and a list of (h, w, d)
    box_centers = np.stack(
        [box_centers_y, box_centers_x, box_centers_z], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_heights, box_widths, box_depths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (y1, x1, y2, x2, z1, z2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    boxes = np.transpose(np.array([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 2], boxes[:, 5]]), axes=(1, 0))
    order = np.array([1, 0, 4, 3, 2, 5])
    all_anchors = boxes[:, order]
    
#     straddle_thresh = cfg.TRAIN.RPN_STRADDLE_THRESH
    straddle_thresh = -1
    
#     cell_anchors = generate_anchors(
#             stride=1. / 16.,
#             scales_xy=((32,64,128,256,512)),
#             scales_z=(32,64,128,256,512),
#             aspect_ratios=(0.25,0.5,1,2,4))
    
#     widths = cell_anchors[:,3] - cell_anchors[:,0]
#     heights = cell_anchors[:,4] - cell_anchors[:,1]
#     depths = cell_anchors[:,5] - cell_anchors[:,2]
#     box_sizes = np.stack([widths, heights, depths]).reshape([-1, 3])

#     all_anchors_l = []
    
#     coords = coords.astype(np.int32)
    
#     for i in coords:
#         all_anchors_l.append(np.hstack((i - 0.5 * box_sizes, i + 0.5 * box_sizes)))

#     all_anchors = np.array(all_anchors_l).reshape(-1, 6)
    
#     coords = coords.astype(np.int32)
#     all_anchors_l = []
#     all_anchors = all_anchors.reshape(-1, 64, 64, 64, 6)
#     for i in coords:
#         all_anchors_l.append(all_anchors[:, i[0], i[1], i[2], :])
        
#     all_anchors = np.array(all_anchors_l).reshape(-1,6)
        
#     all_anchors = all_anchors.reshape(-1, 64, 64, 64, 6)
#     all_anchors = all_anchors.take(np.unique(np.round(coords)[:,0]).astype(np.int32), axis=1).take(np.unique(np.round(coords)[:,1]).astype(np.int32), axis=2).take(np.unique(np.round(coords)[:,2]).astype(np.int32),axis=3)
#     all_anchors = all_anchors.reshape(-1, 6)
    if straddle_thresh > -1:
        # Only keep anchors inside the image by a margin of straddle_thresh
        # Set TRAIN.RPN_STRADDLE_THRESH to -1 (or a large value) to keep all
        # anchors
        inds_inside = np.where(
            (all_anchors[:, 0] >= -straddle_thresh) &
            (all_anchors[:, 1] >= -straddle_thresh) &
            (all_anchors[:, 2] >= -straddle_thresh) &
            (all_anchors[:, 3] < im_width + straddle_thresh) &
            (all_anchors[:, 4] < im_height + straddle_thresh) &
            (all_anchors[:, 5] < im_depth + straddle_thresh)
        )[0]
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
    else:
        t = np.zeros((15, 64, 64, 64))
        for i in np.unique(np.round(coords[:,0])):
            for j in np.unique(np.round(coords[:,1])):
                for k in np.unique(np.round(coords[:,2])):
                    t[:, int(i), int(j), int(k)] = 1.
        inds_inside = t.reshape(-1).nonzero()[0]
#         all_anchors = boxes[:, order]
        anchors = orig_anchors[inds_inside, :]
    num_inside = len(inds_inside)
    
#     logger.debug('total_anchors: %d', total_anchors)
#     logger.debug('inds_inside: %d', num_inside)
#     logger.debug('anchors.shape: %s', str(anchors.shape))

    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside, ), dtype=np.int32)
    labels.fill(-1)
    if len(gt_boxes) > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        time_before = time.time()
        anchor_by_gt_overlap = (box_utils.bbox_overlaps((anchors.astype(dtype=np.float32)), (gt_boxes.astype(dtype=np.float32))))
#         anchor_by_gt_overlap = (box_utils.overlaps_3D(torch.from_numpy(anchors).float(), torch.from_numpy(gt_boxes).float())).numpy()
        print("overlap time: ", time.time() - time_before)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                anchor_to_gt_argmax]

        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax,
            np.arange(anchor_by_gt_overlap.shape[1])
        ]
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max
        )[0]
        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        labels[anchors_with_max_overlap] = 1
        # Fg label: above threshold IOU
#         labels[anchor_to_gt_max >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
#         labels[anchor_to_gt_max >= 0.7] = 1
        labels[anchor_to_gt_max >= 0.45] = 1

    # subsample positive labels if we have too many
#     num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE_PER_IM)
    num_fg = int(0.5 * 512)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False
        )
        labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    # subsample negative labels if we have too many
    # (samples with replacement, but since the set of bg inds is large most
    # samples will not have repeats)
    num_bg = 512 - np.sum(labels == 1)
    bg_inds = np.where(anchor_to_gt_max < 0.01)[0]
    if len(bg_inds) > num_bg:
        enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
        labels[enable_inds] = 0
    bg_inds = np.where(labels == 0)[0]

    bbox_targets = np.zeros((num_inside, 6), dtype=np.float32)
    bbox_targets[fg_inds, :] = compute_targets(anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :])
#     bbox_targets = bbox_targets / np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

    # Bbox regression loss has the form:
    #   loss(x) = weight_outside * L(weight_inside * x)
    # Inside weights allow us to set zero loss on an element-wise basis
    # Bbox regression is only trained on positive examples so we set their
    # weights to 1.0 (or otherwise if config is different) and 0 otherwise
    bbox_inside_weights = np.zeros((num_inside, 6), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    # The bbox regression loss only averages by the number of images in the
    # mini-batch, whereas we need to average by the total number of example
    # anchors selected
    # Outside weights are used to scale each element-wise loss so the final
    # average over the mini-batch is correct
    bbox_outside_weights = np.zeros((num_inside, 6), dtype=np.float32)
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    bbox_outside_weights[labels == 1, :] = 1.0 / num_examples
    bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

    # Map up to original set of anchors
    labels = unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0
    )
    bbox_outside_weights = unmap(
        bbox_outside_weights, total_anchors, inds_inside, fill=0
    )

    # Split the generated labels, etc. into labels per each field of anchors
    blobs_out = []
    start_idx = 0
    for foa in foas:
        H = foa.field_size
        W = foa.field_size
        D = foa.field_size
        A = foa.num_cell_anchors
        end_idx = start_idx + H * W * D * A 
        _labels = labels[start_idx:end_idx]
        _bbox_targets = bbox_targets[start_idx:end_idx, :]
        _bbox_inside_weights = bbox_inside_weights[start_idx:end_idx, :]
        _bbox_outside_weights = bbox_outside_weights[start_idx:end_idx, :]
        start_idx = end_idx

        # labels output with shape (1, A, height, width)
        _labels = _labels.reshape((1, H, W, D, A)).transpose(0, 4, 1, 2, 3)
        # bbox_targets output with shape (1, 4 * A, height, width)
        _bbox_targets = _bbox_targets.reshape(
            (1, H, W, D, A * 6)).transpose(0, 4, 1, 2, 3)
        # bbox_inside_weights output with shape (1, 4 * A, height, width)
        _bbox_inside_weights = _bbox_inside_weights.reshape(
            (1, H, W, D, A * 6)).transpose(0, 4, 1, 2, 3)
        # bbox_outside_weights output with shape (1, 4 * A, height, width)
        _bbox_outside_weights = _bbox_outside_weights.reshape(
            (1, H, W, D, A * 6)).transpose(0, 4, 1, 2, 3)
        blobs_out.append(
            dict(
                rpn_labels_int32_wide=_labels,
                rpn_bbox_targets_wide=_bbox_targets,
                rpn_bbox_inside_weights_wide=_bbox_inside_weights,
                rpn_bbox_outside_weights_wide=_bbox_outside_weights,
                anchors=orig_anchors
            )
        )
    return blobs_out[0] if len(blobs_out) == 1 else blobs_out

def compute_targets(ex_rois, gt_rois, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
    """Compute bounding-box regression targets for an image."""
#     print(box_utils.bbox_transform_inv(ex_rois, gt_rois).sum())
    return box_utils.bbox_transform_inv(ex_rois, gt_rois).astype(
        np.float32, copy=False
    )

def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
        
    return ret

def get_field_of_anchors(
    stride, anchor_sizes, anchor_sizes_z, anchor_aspect_ratios, octave=None, aspect=None
):
    global _threadlocal_foa
    if not hasattr(_threadlocal_foa, 'cache'):
        _threadlocal_foa.cache = {}

#     cache_key = str(stride) + str(anchor_sizes) + str(anchor_sizes_z) + str(anchor_aspect_ratios)
#     if cache_key in _threadlocal_foa.cache:
#         import pdb; pdb.set_trace()
#         return _threadlocal_foa.cache[cache_key]

    # Anchors at a single feature cell
    cell_anchors = generate_anchors(
        stride=stride, scales_xy=anchor_sizes, scales_z=anchor_sizes_z, aspect_ratios=anchor_aspect_ratios
    )
    
    num_cell_anchors = cell_anchors.shape[0]

    # Generate canonical proposals from shifted anchors
    # Enumerate all shifted positions on the (H, W) grid
#     fpn_max_size = cfg.FPN.COARSEST_STRIDE * np.ceil(
#         cfg.TRAIN.MAX_SIZE / float(cfg.FPN.COARSEST_STRIDE)
#     )
    fpn_max_size = 32 * np.ceil(
        1024 / float(32)
    )
    field_size = int(np.ceil((fpn_max_size) / float(stride)))
#     shifts = np.arange(0, field_size) * stride
#     shift_x, shift_y, shift_z = np.meshgrid(shifts, shifts, shifts)

#     coords = input_data[:, :3+1].float()
#     vals = input_data[:, 3+1:].float().cpu().numpy()
#     over_thresh = np.where(vals > 1.)[0]
#     coords = coords[over_thresh, :]
#     coords = (coords/16.).cpu().numpy()

#     shifts_x = np.unique(np.round(coords)[:,0]) * stride
#     shifts_y = np.unique(np.round(coords)[:,1]) * stride
#     shifts_z = np.unique(np.round(coords)[:,2]) * stride

    shifts_x = np.arange(0, field_size, 1) * stride #translate from fm positions to input coords.
    shifts_y = np.arange(0, field_size, 1) * stride
    shifts_z = np.arange(0, field_size, 1) * stride
    shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z)
    
#     import pdb; pdb.set_trace()
    
#     shift_x = shift_x.ravel()
#     shift_y = shift_y.ravel()
#     shift_z = shift_z.ravel()
#     shifts = np.vstack((shifts_x.ravel(), shifts_y.ravel(), shifts_z.ravel(), shifts_x.ravel(), shifts_y.ravel(), shifts_z.ravel())).transpose()

    # Broacast anchors over shifts to enumerate all anchors at all positions
    # in the (H, W) grid:
    #   - add A cell anchors of shape (1, A, 4) to
    #   - K shifts of shape (K, 1, 4) to get
    #   - all shifted anchors of shape (K, A, 4)
    #   - reshape to (K*A, 4) shifted anchors
#     A = num_cell_anchors
#     K = shifts.shape[0]
#     field_of_anchors = (
#         cell_anchors.reshape((1, A, 6)) +
#         shifts.reshape((1, K, 6)).transpose((1, 0, 2))
#     )

    box_widths, box_centers_x = np.meshgrid((cell_anchors[:,3] - cell_anchors[:,0]), shifts_x)
    box_heights, box_centers_y = np.meshgrid((cell_anchors[:,4] - cell_anchors[:,1]), shifts_y)
    box_depths, box_centers_z = np.meshgrid((cell_anchors[:,5] - cell_anchors[:,2]), shifts_z)

    # Reshape to get a list of (y, x, z) and a list of (h, w, d)
    box_centers = np.stack(
        [box_centers_y, box_centers_x, box_centers_z], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_heights, box_widths, box_depths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (y1, x1, y2, x2, z1, z2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    field_of_anchors = np.transpose(np.array([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 2], boxes[:, 5]]), axes=(1, 0))
    
    i = np.array([1, 0, 4, 3, 2, 5])
    field_of_anchors = field_of_anchors[:, i]

#     field_of_anchors = generate_anchors_3D(anchor_sizes, anchor_sizes_z, anchor_aspect_ratios, (65, 65, 65), 16, 16, 16)
#     stride = 16


#     field_of_anchors = field_of_anchors.reshape((K * A, 6))

    foa = FieldOfAnchors(
        field_of_anchors=field_of_anchors.astype(np.float32),
        num_cell_anchors=num_cell_anchors,
        stride=stride,
        field_size=field_size,
        octave=octave,
        aspect=aspect
    )
#     _threadlocal_foa.cache[cache_key] = foa
    return foa

def vis_data(clusters, bbox, vox, cls_segms, ground_truth=False):
    
    cls_segms = [x for x in cls_segms if x != []]
    classes = bbox[:,7]
    scores = bbox[:,6]
    
    keep = np.where(scores > 0.7)[0]
    
#     import pdb; pdb.set_trace()
    
    bbox = bbox[keep, :6]
    classes = classes[keep]
    scores = scores[keep]
    
    layout = go.Layout(
    showlegend=True,
    legend=dict(x=1.01,y=0.95),
    width=1024,
    height=1024,
    hovermode='closest',
    margin=dict(l=0,r=0,b=0,t=0),                                                                                                                                  
    template='plotly_dark',                                                                                                                                        
    uirevision = 'same',
    scene = dict(xaxis = dict(nticks=10, range = (0,1024), showticklabels=True, title='x'),
                 yaxis = dict(nticks=10, range = (0,1024), showticklabels=True, title='y'),
                 zaxis = dict(nticks=10, range = (0,1024), showticklabels=True, title='z'),
                 aspectmode='cube')
    )
    
    from mlreco.visualization import scatter_points, plotly_layout3d
    from mlreco.utils.ppn import uresnet_ppn_type_point_selector
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    from larcv import larcv

    colors = plotly.colors.qualitative.Light24
    
    vox = vox.cpu().numpy()
    # Plot energy depositions (input data)
    trace  = scatter_points(vox,markersize=1.5,color=vox[:,4],colorscale='Jet',
                            cmin=0.01, cmax=1.5,
                            hovertext=['%.2f MeV' % v for v in vox[:,4]])

    # Plot energy depositions (input data)
    colors = plotly.colors.qualitative.Light24

    
# #     if not ground_truth:
#     cls_segms_flat = [j for sub in cls_segms for j in sub]
# #     for i in range(len(cls_segms)):
# #         for j in range(len(cls_segms[i])):
# #             cls_segms_flat.append(np.array(cls_segms[i][j]))

#     cls_segms_flat = np.array(cls_segms_flat)
#     cls_segms_flat = cls_segms_flat[keep]
#     cls_segms_flat = np.concatenate(cls_segms_flat)
    
# #     clusters_masked = []
# #     for i in range(clusters.shape[0]):
# #         if clusters[:, :3][i] in cls_segms_flat:
# #             clusters_masked.append(clusters[i])
# #             print("check!")
            
# #     clusters = np.array(clusters_masked)
# #     cls_segms_flat = np.array(cls_segms_masked)
        
    
# #     segm_tuple = np.array(list(zip(cls_segms_flat[:,0], cls_segms_flat[:,1], cls_segms_flat[:,2])))
# #     cluster_tuple = np.array(list(zip(clusters[:,0], clusters[:,1], clusters[:,2])))
    
#     if cls_segms_flat.shape[0] > 500000:    
#         downsample = np.random.choice(int(cls_segms_flat.shape[0]), 500000, replace=False)
#         cls_segms_flat = cls_segms_flat[downsample, :]
        
# #     segm_tuple = set(list(zip(cls_segms_flat[:,0], cls_segms_flat[:,1], cls_segms_flat[:,2])))
# #     cluster_tuple = set(list(zip(clusters[:,0], clusters[:,1], clusters[:,2])))
# #     intersect = segm_tuple.intersection(cluster_tuple)
    
# #     cls_segms_flat = np.array(list(intersect))
    
# #     color = colors[0]
# #     trace += scatter_points(cls_segms_flat, markersize=1.5, opacity=0.1, color=color)
# #     trace[-1].name = 'cluster %d' % 0
    
#     if not ground_truth:
#         for idx, c in enumerate(np.unique(cls_segms_flat[:,3])):
#             color   = colors[idx % len(colors)]
#             cluster = cls_segms_flat[cls_segms_flat[:,3]==c]
#             trace += scatter_points(cluster, markersize=1.5, color=color)
#             trace[-1].name = 'cluster %d' % idx
#     else:
#         for idx, c in enumerate(np.unique(clusters[:,5])):
#             color   = colors[idx % len(colors)]
#             cluster = clusters[clusters[:,5]==c]
#             trace += scatter_points(cluster,markersize=1.5,color=color)
#             trace[-1].name = 'cluster %d' % idx

    # show
    fig = go.Figure(data=trace,layout=layout)
    
    if not ground_truth:
        for i in range(len(bbox)):
            fig.add_traces(data=scatter_cuboid(bbox[i], scores[i], classes[i], colors=colors[i % len(colors)], opacity=0.2))


    fig.update_layout(legend=dict(x=1.1, y=0.9))
    iplot(fig)
    
def scatter_cuboid(coords, scores, classes, cubesize=1, colors=None, opacity=0.8):
    
    if int(classes) == 1:
        classes = "photon"
    elif int(classes) == 2:
        classes = "electron"
    elif int(classes) == 3:
        classes = "muon"
    elif int(classes) == 4:
        classes = "pion"
    elif int(classes) == 5:
        classes = "protons"
    else:
        classes = "unknown class"
        
    trace = []
    trace.append(go.Mesh3d(
        # 8 vertices of a cube
#             x=[coords[0], coords[0], coords[0], coords[0], coords[3], coords[3], coords[3], coords[3]],
#             y=[coords[1], coords[1], coords[4], coords[4], coords[1], coords[1], coords[4], coords[4]],
#             z=[coords[2], coords[5], coords[2], coords[5], coords[2], coords[5], coords[2], coords[5]],
        x=[coords[0], coords[0], coords[0], coords[0]],
        y=[coords[1], coords[4], coords[1], coords[4]],
        z=[coords[2], coords[2], coords[5], coords[5]],
        opacity=0.2,
        hovertext=str(scores) + ' ' + str(classes),
        color = colors,
        # i, j and k give the vertices of triangles
        delaunayaxis = 'x',
        showscale=True
    ))
    trace.append(go.Mesh3d(
            x=[coords[3], coords[3], coords[3], coords[3]],
            y=[coords[1], coords[4], coords[1], coords[4]],
            z=[coords[2], coords[2], coords[5], coords[5]],
            opacity=0.2,
            hovertext=str(scores) + ' ' + str(classes),
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'x',
            showscale=True
        ))
    trace.append(go.Mesh3d(
            x=[coords[0], coords[3], coords[0], coords[3]],
            y=[coords[1], coords[1], coords[1], coords[1]],
            z=[coords[2], coords[2], coords[5], coords[5]],
            opacity=0.2,
            hovertext=str(scores) + ' ' + str(classes),
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'y',
            showscale=True
        ))
    trace.append(go.Mesh3d(
            x=[coords[0], coords[3], coords[0], coords[3]],
            y=[coords[4], coords[4], coords[4], coords[4]],
            z=[coords[2], coords[2], coords[5], coords[5]],
            opacity=0.2,
            hovertext=str(scores) + ' ' + str(classes),
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'y',
            showscale=True
        ))
    trace.append(go.Mesh3d(
            x=[coords[0], coords[3], coords[0], coords[3]],
            y=[coords[1], coords[1], coords[4], coords[4]],
            z=[coords[2], coords[2], coords[2], coords[2]],
            opacity=0.2,
            hovertext=str(scores) + ' ' + str(classes),
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'z',
            showscale=True
        ))
    trace.append(go.Mesh3d(
            x=[coords[0], coords[3], coords[0], coords[3]],
            y=[coords[1], coords[1], coords[4], coords[4]],
            z=[coords[5], coords[5], coords[5], coords[5]],
            opacity=0.2,
            hovertext=str(scores) + ' ' + str(classes),
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'z',
            showscale=True
        ))
    return trace


def rpn_losses(rpn_cls_logits, rpn_bbox_pred, rpn_labels_int32_wide, rpn_bbox_targets_wide, rpn_bbox_inside_weights_wide, rpn_bbox_outside_weights_wide, anchors, input_data, gt_boxes):
    
    rpn_labels_int32_wide, rpn_bbox_targets_wide, rpn_bbox_inside_weights_wide, rpn_bbox_outside_weights_wide = get_targets(rpn_cls_logits.cpu().detach(), rpn_bbox_pred.cpu().detach(), anchors, gt_boxes, 1024, 1024, 1024)
    
    h, w, d = rpn_cls_logits.shape[2:]
    rpn_labels_int32 = torch.from_numpy(rpn_labels_int32_wide[:, :, :h, :w, :d])  # -1 means ignore
    h, w, d = rpn_bbox_pred.shape[2:]
    rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets_wide[:, :, :h, :w, :d])
    rpn_bbox_inside_weights = torch.from_numpy(rpn_bbox_inside_weights_wide[:, :, :h, :w, :d])
    rpn_bbox_outside_weights = torch.from_numpy(rpn_bbox_outside_weights_wide[:, :, :h, :w, :d])

    # rpn_labels_int32[rpn_labels_int32 == -1] = 0
#     weight_fg = (rpn_labels_int32 == 1) * ((rpn_labels_int32 == 0).sum() / (rpn_labels_int32 == 1).sum())
#     weight_bg = (rpn_labels_int32 == 0).float()
    
#     weight = weight_fg + weight_bg
    weight = (rpn_labels_int32 >= 0).float()
#     weight2 = (rpn_cls_logits > 0).float()
#     weight = weight * weight2.cpu()
    
    loss_rpn_cls = F.binary_cross_entropy_with_logits(rpn_cls_logits.cpu(), rpn_labels_int32.float(), weight, reduction='sum')
    loss_rpn_cls /= weight.sum()
    
#     weight_bbox = (rpn_bbox_pred > 0).float()
#     rpn_bbox_outside_weights = rpn_bbox_outside_weights * weight_bbox.cpu()
    loss_rpn_bbox_l1 = net_utils.smooth_l1_loss(rpn_bbox_pred.cpu(), rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, beta=1/9)
#     loss_rpn_bbox_diou = net_utils.compute_diou(rpn_bbox_pred.cpu(), rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
    loss_rpn_bbox = loss_rpn_bbox_l1
#     print("RPN IOU: ", loss_rpn_bbox_diou)
    

    
#     indicies = torch.where(reshape_format(rpn_bbox_inside_weights) != 0)[0]
#     rpn_bbox_pred_re = reshape_format(rpn_bbox_pred.cpu().double())[indicies]
#     rpn_bbox_targets_re = reshape_format(rpn_bbox_targets.double())[indicies]
# #     loss_rpn_bbox = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets)
#     print("preds: ")
#     print(rpn_bbox_pred_re.max())
#     print(rpn_bbox_pred_re.min())
#     print("targets: ")
#     print(rpn_bbox_targets_re.max())
#     print(rpn_bbox_targets_re.min())
#     print("diff: ")
#     print(torch.abs(rpn_bbox_targets_re - rpn_bbox_pred_re).mean())
    
# #     loss_rpn_bbox = net_utils.compute_diou(rpn_bbox_pred.cpu(), rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
    
# #     if loss_rpn_bbox > 0.5:
# #         import pdb; pdb.set_trace()
    
#     cls_np = rpn_cls_logits.cpu()
#     rpn_cls_probs = torch.sigmoid(cls_np)
    
#     rpn_cls_probs_w = rpn_cls_probs
#     rpn_cls_probs_t = rpn_cls_probs_w.reshape((rpn_cls_probs_w.shape[1], rpn_cls_probs_w.shape[2], rpn_cls_probs_w.shape[3], rpn_cls_probs_w.shape[4]))
#     scores = rpn_cls_probs_t.permute((1, 2, 3, 0)).reshape((-1, 1))
    
#     gt_np = rpn_labels_int32.cpu()
#     gt_cls = rpn_labels_int32[np.where(rpn_labels_int32 >= 0)]
    
#     weight_t = weight.reshape((rpn_cls_probs.shape[1], rpn_cls_probs.shape[2], rpn_cls_probs.shape[3], rpn_cls_probs.shape[4]))
#     weight = weight_t.numpy().transpose((1, 2, 3, 0)).reshape((-1, 1))
#     scores = scores[np.where(weight > 0)]

# #     accu_rpn_cls = 1 - ((torch.abs(gt_cls - scores.float())).sum() / float(gt_cls.shape[0]))
    accu_rpn_cls = 0.
    
#     diff = (gt_cls - scores)
    
#     loss_rpn_cls = 5 * F.mse_loss(scores, gt_cls.float())

#     rpn_bbox_pred_t = rpn_bbox_pred[0, :, :, :, :]
#     bbox_deltas = rpn_bbox_pred_t.data.cpu().numpy()
#     bbox_deltas_re = bbox_deltas.transpose((1, 2, 3, 0)).reshape((-1, 6))
#     all_proposals_pre = box_utils.bbox_transform(anchors, bbox_deltas_re, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
#     all_proposals = box_utils.clip_tiled_boxes(all_proposals_pre, (1024, 1024, 1024))
# #     filters = _filter_boxes(all_proposals, 0, (1024, 1024, 1024, 1))
# #     all_proposals = all_proposals[filters, :]
    
# #     bbox_pred = reshape_format(rpn_bbox_pred)
#     bbox_targets = reshape_format(rpn_bbox_targets)
#     all_gt = box_utils.bbox_transform(anchors, bbox_targets.cpu().numpy(), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
#     weights_i = reshape_format(rpn_bbox_inside_weights)
#     weights_o = reshape_format(rpn_bbox_outside_weights)
    
#     loss_rpn_bbox_diou = net_utils.compute_diou(torch.from_numpy(all_proposals), torch.from_numpy(all_gt), weights_i, weights_o)
#     loss_rpn_bbox =  loss_rpn_bbox_diou + 0.5 * loss_rpn_bbox_l1
#     print("DISTANCE: ", loss_rpn_bbox_diou)

#     loss_rpn_bbox = net_utils.compute_diou(torch.from_numpy(all_proposals), torch.from_numpy(all_gt), weights_i, weights_o)
    
# #     scores = scores[filters]
    
# #     inds = np.argpartition(-scores.squeeze(),
# #                            192000)[:192000]
# #     order = np.argsort(-scores[inds].squeeze())
# #     order = inds[order]
    
# #     scores = scores[order]
# #     all_proposals = all_proposals[order]
    

#     rpn_bbox_outside_weights_t = rpn_bbox_outside_weights[0, :, :, :, :]
#     weights_o = rpn_bbox_outside_weights_t.data.cpu().numpy()
#     weights_r = weights_o.transpose((1, 2, 3, 0)).reshape((-1, 6))
#     weights_f = weights_r[filters, :]
#     weighted_prop = (all_proposals * (weights_f > 0))
#     proposals = weighted_prop[np.where(weighted_prop.sum(1) != 0)]
    
#     keep = np.where(np.logical_and.reduce((proposals[:, 0] > 0, proposals[:, 3] < 350, proposals[:, 1] > 550, proposals[:, 4] < 875, proposals[:, 2] > 450, proposals[:, 5] < 900)))[0]
#     print(keep)
#     boxes = proposals
#     scores_v = scores.detach().cpu().numpy().reshape(scores.shape[0], 1)
# #     scores_v = scores[:30].detach().cpu().numpy().reshape(30, 1)

#     gt_class = np.zeros((boxes.shape[0], 1))

#     cls_boxes = np.hstack((boxes, scores_v, gt_class))

#     vis_data([], cls_boxes, input_data, [])
#     import pdb; pdb.set_trace()

    return loss_rpn_cls, loss_rpn_bbox, accu_rpn_cls

def get_targets(rpn_cls_pred, rpn_bbox_pred, all_anchors, gt_boxes, im_width, im_height, im_depth):
    coords = rpn_cls_pred.nonzero()[:, 1:]
    coords_bbox = rpn_bbox_pred.nonzero()[:, 1:]
#     vals = input_data[:, 3+1:].float().cpu().numpy()
#     over_thresh = np.where(vals > 1.)[0]
#     coords = coords[over_thresh, :]
#     coords = (coords/16.)[:, :3]
    
    cell_anchors = generate_anchors(
            stride=16.,
            scales_xy=((32,64,128,256,512)),
            scales_z=(32,64,128,256,512),
            aspect_ratios=(0.5,1,2))
    
    total_anchors = all_anchors.shape[0]
    
    shift_x = np.unique(coords[:,0] * 16.)
    shift_y = np.unique(coords[:,1] * 16.)
    shift_z = np.unique(coords[:,2] * 16.)

    shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z, copy=False)

    num_images = 1

    box_widths, box_centers_x = np.meshgrid(cell_anchors[:,3] - cell_anchors[:,0], shift_x)
    box_heights, box_centers_y = np.meshgrid(cell_anchors[:,4] - cell_anchors[:,1], shift_y)
    box_depths, box_centers_z = np.meshgrid(cell_anchors[:,5] - cell_anchors[:,2], shift_z)

    # Reshape to get a list of (y, x, z) and a list of (h, w, d)
    box_centers = np.stack(
        [box_centers_y, box_centers_x, box_centers_z], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_heights, box_widths, box_depths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (y1, x1, y2, x2, z1, z2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    boxes = np.transpose(np.array([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 2], boxes[:, 5]]), axes=(1, 0))
    order = np.array([1, 0, 4, 3, 2, 5])
    
    straddle_thresh = -1
    
    if straddle_thresh > -1:
        # Only keep anchors inside the image by a margin of straddle_thresh
        # Set TRAIN.RPN_STRADDLE_THRESH to -1 (or a large value) to keep all
        # anchors
        inds_inside = np.where(
            (all_anchors[:, 0] >= -straddle_thresh) &
            (all_anchors[:, 1] >= -straddle_thresh) &
            (all_anchors[:, 2] >= -straddle_thresh) &
            (all_anchors[:, 3] < im_width + straddle_thresh) &
            (all_anchors[:, 4] < im_height + straddle_thresh) &
            (all_anchors[:, 5] < im_depth + straddle_thresh)
        )[0]
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
    else:
#         inds_inside = np.arange(all_anchors.shape[0])
        t = np.zeros((15, 64, 64, 64))
#         for i in np.unique(coords[:,0]):
#             for j in np.unique(coords[:,1]):
#                 for k in np.unique(coords[:,2]):
#                     t[:, i, j, k] = 1.
        for i in coords:
            t[i[0], i[1], i[2], i[3]] = 1.
        inds_inside = t.reshape(-1).nonzero()[0]
#         inds_inside_bbox = rpn_bbox_pred.reshape(-1,6).nonzero()[:,0].unique()[torch.where(rpn_bbox_pred.reshape(-1,6).nonzero()[:,0].unique(return_counts=True)[1] == 6)[0].numpy()]
#         inds_inside = np.hstack((inds_inside, inds_inside_bbox))
#         anchors_bbox = all_anchors[inds_inside_bbox, :]
        all_anchors = all_anchors[inds_inside, :]
        anchors = all_anchors
    num_inside = len(np.arange(all_anchors.shape[0]))
    
#     logger.debug('total_anchors: %d', total_anchors)
#     logger.debug('inds_inside: %d', num_inside)
#     logger.debug('anchors.shape: %s', str(anchors.shape))

    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside, ), dtype=np.int32)
    labels.fill(-1)
    if len(gt_boxes) > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        time_before = time.time()
        anchor_by_gt_overlap = (box_utils.bbox_overlaps((anchors.astype(dtype=np.float32)), (gt_boxes.astype(dtype=np.float32))))
#         anchor_by_gt_overlap_bbox = (box_utils.bbox_overlaps((anchors_bbox.astype(dtype=np.float32)), (gt_boxes.astype(dtype=np.float32))))
#         anchor_by_gt_overlap = (box_utils.overlaps_3D(torch.from_numpy(anchors).float(), torch.from_numpy(gt_boxes).float())).numpy()
        print("overlap time: ", time.time() - time_before)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
#         anchor_to_gt_argmax_bbox = anchor_by_gt_overlap_bbox.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                anchor_to_gt_argmax]

        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax,
            np.arange(anchor_by_gt_overlap.shape[1])
        ]
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max
        )[0]
        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        labels[anchors_with_max_overlap] = 1
        # Fg label: above threshold IOU
#         labels[anchor_to_gt_max >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
#         labels[anchor_to_gt_max >= 0.7] = 1
        labels[anchor_to_gt_max >= 0.1] = 1

    # subsample positive labels if we have too many
#     num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE_PER_IM)
    num_fg = int(0.5 * 512)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False
        )
        labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    # subsample negative labels if we have too many
    # (samples with replacement, but since the set of bg inds is large most
    # samples will not have repeats)
    num_bg = 512 - np.sum(labels == 1)
    bg_inds = np.where(anchor_to_gt_max < 0.01)[0]
    if len(bg_inds) > num_bg:
        enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
        labels[enable_inds] = 0
    bg_inds = np.where(labels == 0)[0]

    bbox_targets = np.zeros((num_inside, 6), dtype=np.float32)
#     bbox_targets[inds_inside_bbox, :] = compute_targets(anchors_bbox, gt_boxes[anchor_to_gt_argmax_bbox, :])
    bbox_targets[fg_inds, :] = compute_targets(anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :])
#     bbox_targets = bbox_targets / np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

    # Bbox regression loss has the form:
    #   loss(x) = weight_outside * L(weight_inside * x)
    # Inside weights allow us to set zero loss on an element-wise basis
    # Bbox regression is only trained on positive examples so we set their
    # weights to 1.0 (or otherwise if config is different) and 0 otherwise
    bbox_inside_weights = np.zeros((num_inside, 6), dtype=np.float32)
    
#     temp = np.full((inds_inside_bbox.shape[0] - labels.shape[0]), -1)
#     tlabel = np.hstack((labels, temp))
    bbox_inside_weights[labels == 1, :] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#     bbox_inside_weights[inds_inside_bbox, :] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    # The bbox regression loss only averages by the number of images in the
    # mini-batch, whereas we need to average by the total number of example
    # anchors selected
    # Outside weights are used to scale each element-wise loss so the final
    # average over the mini-batch is correct
    bbox_outside_weights = np.zeros((num_inside, 6), dtype=np.float32)
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
#     bbox_outside_weights[inds_inside_bbox, :] = 1.0 / num_examples
    bbox_outside_weights[labels == 1, :] = 1.0 / num_examples
    bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

    # Map up to original set of anchors
    labels = unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0
    )
    bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
    
    # Split the generated labels, etc. into labels per each field of anchors
    blobs_out = []
    start_idx = 0
    H = 64
    W = 64
    D = 64
    A = 15
    end_idx = start_idx + H * W * D * A
    _labels = labels[start_idx:end_idx]
    _bbox_targets = bbox_targets[start_idx:end_idx, :]
    _bbox_inside_weights = bbox_inside_weights[start_idx:end_idx, :]
    _bbox_outside_weights = bbox_outside_weights[start_idx:end_idx, :]
    start_idx = end_idx
    # labels output with shape (1, A, height, width)
    _labels = _labels.reshape((1, A, H, W, D))
    # bbox_targets output with shape (1, 4 * A, height, width)
#     _bbox_targets = _bbox_targets.reshape((1,A*6,64,64,64))

    _bbox_targets = _bbox_targets.reshape((1, A, 64, 64, 64, 6)).transpose(0,1,5,2,3,4).reshape(1, -1, 64, 64, 64)
    _bbox_inside_weights = _bbox_inside_weights.reshape((1, A, 64, 64, 64, 6)).transpose(0,1,5,2,3,4).reshape(1, -1, 64, 64, 64)
    _bbox_outside_weights = _bbox_outside_weights.reshape((1, A, 64, 64, 64, 6)).transpose(0,1,5,2,3,4).reshape(1, -1, 64, 64, 64)
    # bbox_inside_weights output with shape (1, 4 * A, height, width)
#     _bbox_inside_weights = _bbox_inside_weights.reshape((1,A*6,64,64,64))
    # bbox_outside_weights output with shape (1, 4 * A, height, width)
#     _bbox_outside_weights = _bbox_outside_weights.reshape((1,A*6,64,64,64))
    
    return _labels, _bbox_targets, _bbox_inside_weights, _bbox_outside_weights

def reshape_format(a):
    return a[0, :, :, :, :].permute(1, 2, 3, 0).reshape((-1, 6))