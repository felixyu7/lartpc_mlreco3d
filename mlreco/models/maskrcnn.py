import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import mlreco.models.sparse_resnet as sparse_resnet
import mlreco.models.RPN as RPN
import scipy
import mlreco.models.utils.boxes as box_utils
import mlreco.models.utils.blob as blob_utils
from mlreco.models.roi_align import roi_align
import mlreco.models.fast_rcnn_heads as fast_rcnn_heads
import mlreco.models.mask_rcnn_heads as mask_rcnn_heads
import mlreco.models.utils.boxes as box_utils
import torch.nn.functional as F

from scipy.ndimage import zoom
import pycocotools.mask as mask_util

import plotly.graph_objs as go
import plotly
import time


class MaskRCNN(nn.Module):

    def __init__(self, cfg, name="maskrcnn"):
        super(MaskRCNN,self).__init__()

        self.dim_out = 256
        self.backbone = sparse_resnet.SparseResNet(cfg)
        self.RPN = RPN.RPN(cfg)
        self.Box_Head = sparse_resnet.ResNet_roi_conv5_head(int(self.dim_out/2), self.roi_feature_transform, 1/16, dim_out=self.dim_out*2)
        self.Box_Class = fast_rcnn_heads.fast_rcnn_outputs(self.dim_out*2)
        self.Mask_Head = mask_rcnn_heads.mask_rcnn_fcn_head_v0upshare(self.RPN.dim_out, self.roi_feature_transform, 1/16)
        self.Mask_Head.share_res5_module(self.Box_Head.res5)
        self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

    def forward(self, input):
        #print(input[1][0].shape)
        print("shape input data", input[0].shape)
        return_dict = {}

        input_data = input[0]

        before_resnet = time.time()
        feature_map = self.backbone(input_data)
        print("Time taken in ResNet: ", time.time() - before_resnet)

#         if not self.training:
#             return_dict['blob_conv'] = feature_map

        roidb = construct_roidb(input)

        before_construct = time.time()
        blobs, valid = get_minibatch(roidb)
        print("Time taken to construct roidb: ", time.time() - before_construct)

        before_rpn = time.time()
        rpn_ret = self.RPN(feature_map, torch.from_numpy(blobs['im_info']), blobs['roidb'])
        print("Time taken in RPN: ", time.time() - before_rpn)

        before_boxhead = time.time()
        if self.training:
            box_head_ret, res5_feat = self.Box_Head(feature_map, rpn_ret)
        else:
            box_head_ret = self.Box_Head(feature_map, rpn_ret)
        print("Time take in Box Head: ", time.time() - before_boxhead)

        cls_score, bbox_pred = self.Box_Class(box_head_ret)

#         if self.training:
#             mask_feat = self.Mask_Head(res5_feat, rpn_ret, roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
#             mask_pred = self.Mask_Outs(mask_feat)

        if self.training:
            res = {'labels_int32': [rpn_ret['labels_int32']],
                   'bbox_targets': [rpn_ret['bbox_targets']],
                   'bbox_inside_weights': [rpn_ret['bbox_inside_weights']],
                   'bbox_outside_weights': [rpn_ret['bbox_outside_weights']],
                   'rpn_cls_logits': [rpn_ret['rpn_cls_logits']],
                   'rpn_bbox_pred': [rpn_ret['rpn_bbox_pred']],
                   'rpn_labels_int32_wide': [blobs['rpn_labels_int32_wide']],
                   'rpn_bbox_targets_wide': [blobs['rpn_bbox_targets_wide']],
                   'rpn_bbox_inside_weights_wide': [blobs['rpn_bbox_inside_weights_wide']],
                   'rpn_bbox_outside_weights_wide': [blobs['rpn_bbox_outside_weights_wide']],
                   'cls_score': [cls_score],
                   'bbox_pred': [bbox_pred],
                   'bbox_proposals': [rpn_ret['rpn_rois']],
                   'bbox_scores': [rpn_ret['rpn_roi_probs']]}
#                    'masks_int32': [rpn_ret['masks_int32']],
#                    'mask_pred': [mask_pred]}
#             res = {'rpn_cls_logits': [rpn_ret['rpn_cls_logits']],
#                    'rpn_bbox_pred': [rpn_ret['rpn_bbox_pred']],
#                    'rpn_labels_int32_wide': [blobs['rpn_labels_int32_wide']],
#                    'rpn_bbox_targets_wide': [blobs['rpn_bbox_targets_wide']],
#                    'rpn_bbox_inside_weights_wide': [blobs['rpn_bbox_inside_weights_wide']],
#                    'rpn_bbox_outside_weights_wide': [blobs['rpn_bbox_outside_weights_wide']]}
# #                    'cls_score': [cls_score],
# #                    'bbox_pred': [bbox_pred],
# #                    'bbox_proposals': [rpn_ret['rpn_rois']],
# #                    'bbox_scores': [rpn_ret['rpn_roi_probs']],
# #                    'masks_int32': [rpn_ret['masks_int32']],
# #                    'mask_pred': [mask_pred]}
        else:
            scores, pred_boxes = im_detect_bbox(rpn_ret['rois'], bbox_pred, cls_score)
            scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, pred_boxes)

            # masks = im_detect_mask(self, boxes, feature_map)
            # cls_segms, round_boxes = segm_results(cls_boxes, masks, boxes, 1024, 1024, 1024, roidb[0]['clusters'])

            cls_segms = []
            cls_boxes = prep_pred_vis(cls_boxes)
            vis_data(roidb[0]['clusters'], cls_boxes, input_data, cls_segms)

            gt_cls_boxes = prep_gt_vis(input[1][0].cpu().numpy().astype(float))
            vis_data(roidb[0]['clusters'], gt_cls_boxes, input_data, cls_segms, ground_truth=True)
            import pdb; pdb.set_trace()

        return res

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        # Single feature level
        # rois: holds R regions of interest, each is a 5-tuple
        # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
        # rectangle (x1, y1, x2, y2)
        device_id = ''
        if blobs_in.is_cuda:
            device_id = blobs_in.get_device()
        else:
            device_id = 'cpu'
        #print(device_id)
        rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).to(torch.device(device_id))
        if method == 'RoIPoolF':
            xform_out = ROIPool((resolution, resolution), spatial_scale)(blobs_in, rois)
        elif method == 'RoIAlign':
            i = [0, 1, 2, 4, 5, 3, 6]
            rois = rois[:, i]
            xform_out = roi_align.RoIAlign(
                (resolution, resolution, resolution), spatial_scale, sampling_ratio)(blobs_in, rois.float())

        return xform_out

def construct_roidb(input):

    bbox_out = input[1][0].cpu().numpy().astype(float)

    box_areas = (bbox_out[:,3] - bbox_out[:,0]) * (bbox_out[:,4] - bbox_out[:,1]) * (bbox_out[:,5] - bbox_out[:,2])

    roidb = {}
    roidb['boxes'] = bbox_out[:,:6].astype(float)

    roidb['clusters'] = input[2].cpu().numpy().astype(float)
    roidb['clusters'][:,4] *= 100

    ids = bbox_out[:, 7]
    gids = bbox_out[:, 8]
    shower = bbox_out[:, 9]

    masks = []

    for i in range(len(shower)):
        if shower[i] == 0:
            masks.append(roidb['clusters'][np.where(roidb['clusters'][:,5] == ids[i])][:, :3])
        else:
            masks.append(roidb['clusters'][np.where(roidb['clusters'][:,6] == gids[i])][:, :3])

    roidb['segms'] = masks

    roidb['gt_classes'] = (bbox_out[:,6].reshape(len(roidb['boxes'])) + 1).astype(int)

    # compensate for background '0' class
    roidb['gt_classes'] += 1

    gt_overlaps = np.zeros((len(roidb['boxes']), 6)).astype(float)
    box_to_gt_ind_map = np.zeros((len(roidb['boxes']))).astype(int)

    for x in range(roidb['gt_classes'].shape[0]):
        gt_overlaps[x][roidb['gt_classes'][x]] = 1
        box_to_gt_ind_map[x] = x

    roidb['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
    roidb['box_to_gt_ind_map'] = box_to_gt_ind_map
    roidb['seg_areas'] = np.zeros((len(roidb['boxes'])))

    return [roidb]

def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list

    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales = _get_image_blob(roidb)
    blobs['data'] = im_blob

    valid = RPN.add_rpn_blobs(blobs, im_scales, roidb)

    return blobs, valid

def prep_images(clusters):

    coords = clusters[:,:3].astype(int)
    values = clusters[:,4]

    im = np.zeros((1024, 1024, 1024, 3), dtype=np.float32)

    for i in range(len(coords)):
        im[coords[i][0]][coords[i][1]][coords[i][2]][0] = float(values[i])
        im[coords[i][0]][coords[i][1]][coords[i][2]][1] = float(values[i])
        im[coords[i][0]][coords[i][1]][coords[i][2]][2] = float(values[i])

    return im

def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=1, size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        t_st = time.time()
        im = prep_images(roidb[i]['clusters'])
        print("prep images: ", time.time() - t_st)
        target_size = 1024
        t_st = time.time()
        im = [im]
        im_scale = [1.]
#         im, im_scale = blob_utils.prep_im_for_blob(
#             im, np.array([[[0.0, 0.0, 0.0]]]), [target_size], 1024)
        print("prep blob: ", time.time() - t_st)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]

    t_st = time.time()
    blob = blob_utils.im_list_to_blob(processed_ims)
    print("list to blob: ", time.time() - t_st)

    return blob, im_scales

def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    blob_names += RPN.get_rpn_blob_names(is_training=is_training)

    return blob_names

class MaskRCNN_Loss(nn.Module):

    def __init__(self, cfg, name="maskrcnn_loss"):
        super(MaskRCNN_Loss, self).__init__()
        self.model_cfg = cfg[name]
        self.weight_rpn_cls = self.model_cfg.get('weight_rpn_cls', 1.)
        self.weight_rpn_bbox = self.model_cfg.get('weight_rpn_bbox', 1.)
        self.weight_cls = self.model_cfg.get('weight_cls', 1.)
        self.weight_bbox = self.model_cfg.get('weight_bbox', 1.)
        print("weights:", self.weight_rpn_cls, self.weight_rpn_bbox, self.weight_cls, self.weight_bbox)

    def forward(self, out, clusters, input_data):

        return_dict = {}

        loss_rpn_cls, loss_rpn_bbox = RPN.rpn_losses(out['rpn_cls_logits'][0], out['rpn_bbox_pred'][0], out['rpn_labels_int32_wide'][0], out['rpn_bbox_targets_wide'][0], out['rpn_bbox_inside_weights_wide'][0], out['rpn_bbox_outside_weights_wide'][0])

        loss_cls, loss_bbox, accuracy_cls, accuracy_neut, accuracy_cosm = fast_rcnn_heads.fast_rcnn_losses(
        out['cls_score'][0], out['bbox_pred'][0], out['labels_int32'][0], out['bbox_targets'][0],
        out['bbox_inside_weights'][0], out['bbox_outside_weights'][0])

#         loss_mask, mask_accuracies = mask_rcnn_heads.mask_rcnn_losses(out['mask_pred'][0], out['masks_int32'][0])

        loss_rpn_cls = loss_rpn_cls.cuda()
        loss_rpn_bbox = loss_rpn_bbox.cuda()
        loss_cls = loss_cls.cuda()
        loss_bbox = loss_bbox.cuda()
#         loss_mask = loss_mask.cuda()

#         loss = loss_rpn_cls + loss_rpn_bbox + loss_cls + loss_bbox + loss_mask
        loss = self.weight_rpn_cls * loss_rpn_cls + self.weight_rpn_bbox * loss_rpn_bbox + self.weight_cls * loss_cls + self.weight_bbox *  loss_bbox
#         loss = loss_mask

        return_dict['loss_rpn_cls'] = self.weight_rpn_cls * loss_rpn_cls
        return_dict['loss_rpn_bbox'] = self.weight_rpn_bbox * loss_rpn_bbox
        return_dict['loss_cls'] = self.weight_cls * loss_cls
        return_dict['loss_bbox'] = self.weight_bbox * loss_bbox
#         return_dict['loss_mask'] = loss_mask
        return_dict['accuracy_cls'] = accuracy_cls
#         return_dict['accuracy_mask'] = mask_accuracies[6]
        return_dict['loss'] = loss
        return_dict['accuracy'] = accuracy_cls
#         return_dict['accuracy_mask_photon'] = mask_accuracies[1]
#         return_dict['accuracy_mask_electron'] = mask_accuracies[2]
#         return_dict['accuracy_mask_muon'] = mask_accuracies[3]
#         return_dict['accuracy_mask_pion'] = mask_accuracies[4]
#         return_dict['accuracy_mask_proton'] = mask_accuracies[5]
#         return_dict['pred_on_A'] = mask_accuracies[7]
#         return_dict['B_pred_A'] = mask_accuracies[8]
#         return_dict['A_pred_B'] = mask_accuracies[9]

#         print(loss_rpn_bbox)
#         print(loss_bbox)
#         scores, pred_boxes = im_detect_bbox(out['bbox_proposals'][0], out['bbox_pred'][0], out['cls_score'][0])
#         vis_data(clusters, pred_boxes, scores, input_data)

        return return_dict


# Inference functions

def im_detect_bbox(bbox_proposals, bbox_pred, cls_score):

    cls_score = F.softmax(cls_score, dim=1)

    scores = cls_score.data.cpu().numpy().squeeze()
    scores = scores.reshape([-1, scores.shape[-1]])
    scores_nobk = scores[:,1:]
    score_max = scores_nobk.max(1)

    boxes = bbox_proposals[:, 1:]

    box_deltas = bbox_pred.data.cpu().numpy().squeeze()
    # In case there is 1 proposal
    box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])

    pred_boxes = box_utils.bbox_transform(boxes, box_deltas, (10., 10., 10., 5., 5., 5.))
    pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, [1024, 1024, 1024])

    return scores, pred_boxes

def box_results_with_nms_and_limit(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = 6
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > 0.05)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 6:(j + 1) * 6]
#         dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)

        keep = box_utils.nms(boxes_j, scores_j, 0.3)
        keep = keep.cpu().numpy()
        boxes_j = boxes_j[keep, :]

#         box_ids = np.arange(last, last + boxes_j.shape[0])
#         boxes_j = np.hstack((boxes_j, box_ids))
#         last = last + boxes_j.shape[0] + 1

        scores_j = scores_j[keep]
        nms_dets = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)

        # Refine the post-NMS boxes using bounding-box voting
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    image_scores = np.hstack(
        [cls_boxes[j][:, -1] for j in range(1, num_classes)]
    )
    if len(image_scores) > 100:
        image_thresh = np.sort(image_scores)[-100]
        for j in range(1, num_classes):
            keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
            cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes

def im_detect_mask(self, boxes, feature_map):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scale (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)
        blob_conv (Variable): base features from the backbone network.

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = 192

    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, 1.)}

    mask_feat = self.Mask_Head(feature_map, inputs)
    pred_masks = self.Mask_Outs(mask_feat)
    pred_masks = pred_masks.data.cpu().numpy().squeeze()

    pred_masks = pred_masks.reshape([-1, 6, M, M, M])

    return pred_masks

def segm_results(cls_boxes, masks, ref_boxes, im_w, im_h, im_d, clusters, polygon=True):
    num_classes = 6
    cls_segms = [[] for _ in range(num_classes)]
    round_boxes =[[] for _ in range(num_classes)]
    mask_ind = 0


    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = 192
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2, M + 2), dtype=np.float32)
    # skip j = 0, because it's the background class

    gt_im = np.zeros((1024, 1024, 1024), dtype=np.uint8)
    for i in clusters:
        gt_im[int(i[0])][int(i[1])][int(i[2])] = 1

    cluster_id = 0
    for j in range(1, num_classes):
        segms = []
        rboxes= []
        for _ in range(cls_boxes[j].shape[0]):

            padded_mask[1:-1, 1:-1, 1:-1] = masks[mask_ind, j, :, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = (ref_box[3] - ref_box[0] + 1)
            h = (ref_box[4] - ref_box[1] + 1)
            d = (ref_box[5] - ref_box[2] + 1)
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)
            d = np.maximum(d, 1)

#             mask = cv2.resize(padded_mask, (w, h))
            orig_shape = padded_mask.shape
            mask = zoom(padded_mask, (w/orig_shape[0], h/orig_shape[1], d/orig_shape[2]), order=1)
            mask = np.array(mask > 0.5, dtype=np.uint8)
            im_mask = np.zeros((im_w, im_h, im_d), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[3] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[4] + 1, im_h)
            z_0 = max(ref_box[2], 0)
            z_1 = min(ref_box[5] + 1, im_d)

#             # Get RLE encoding used by the COCO evaluation API
#             im_mask[x_0:x_1, y_0:y_1, z_0:z_1] = mask[
#                 (x_0 - ref_box[0]):(x_1 - ref_box[0]), (y_0 - ref_box[1]):(y_1 - ref_box[1]), (z_0 - ref_box[2]):(z_1 - ref_box[2])]

#             rle = mask_util.encode(np.array(im_mask[:, :, :, np.newaxis], order='F'))[0]
#             # For dumping to json, need to decode the byte string.
#             # https://github.com/cocodataset/cocoapi/issues/70
#             rle['counts'] = rle['counts'].decode('ascii')
#             segms.append(rle)
            mask[(x_0 - ref_box[0]):(x_1 - ref_box[0]), (y_0 - ref_box[1]):(y_1 - ref_box[1]), (z_0 - ref_box[2]):(z_1 - ref_box[2])] *= gt_im[x_0:x_1, y_0:y_1, z_0:z_1]
            non_zero_mask = np.transpose(np.nonzero(mask[
                    (x_0 - ref_box[0]):(x_1 - ref_box[0]),
                    (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                    (z_0 - ref_box[2]):(z_1 - ref_box[2])
                    ]))
            non_zero_mask[:,0] += x_0
            non_zero_mask[:,1] += y_0
            non_zero_mask[:,2] += z_0
#             vals = np.ones(non_zero_mask.shape[1])
#             mask_size = mask[
#                     (x_0 - ref_box[0]):(x_1 - ref_box[0]),
#                     (y_0 - ref_box[1]):(y_1 - ref_box[1]),
#                     (z_0 - ref_box[2]):(z_1 - ref_box[2])
#                     ].shape
            sparse_mask = non_zero_mask
            cluster_ids = np.full((sparse_mask.shape[0], 1), cluster_id)
            sparse_mask = np.hstack((sparse_mask, cluster_ids))
            segms.append(sparse_mask)
            rboxes.append(ref_box)
            cluster_id += 1
            mask_ind += 1

        cls_segms[j] = segms
        round_boxes[j] = rboxes
    assert mask_ind == masks.shape[0]

    return cls_segms, round_boxes

def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels

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

    start_box = time.time()

#     if not ground_truth:
    cls_segms_flat = [j for sub in cls_segms for j in sub]
#     for i in range(len(cls_segms)):
#         for j in range(len(cls_segms[i])):
#             cls_segms_flat.append(np.array(cls_segms[i][j]))

    cls_segms_flat = np.array(cls_segms_flat)
    cls_segms_flat = cls_segms_flat[keep]
    if len(cls_segms_flat) > 0:
        cls_segms_flat = np.concatenate(cls_segms_flat)

#     clusters_masked = []
#     for i in range(clusters.shape[0]):
#         if clusters[:, :3][i] in cls_segms_flat:
#             clusters_masked.append(clusters[i])
#             print("check!")

#     clusters = np.array(clusters_masked)
#     cls_segms_flat = np.array(cls_segms_masked)


#     segm_tuple = np.array(list(zip(cls_segms_flat[:,0], cls_segms_flat[:,1], cls_segms_flat[:,2])))
#     cluster_tuple = np.array(list(zip(clusters[:,0], clusters[:,1], clusters[:,2])))

    if cls_segms_flat.shape[0] > 500000:
        downsample = np.random.choice(int(cls_segms_flat.shape[0]), 500000, replace=False)
        cls_segms_flat = cls_segms_flat[downsample, :]

#     segm_tuple = set(list(zip(cls_segms_flat[:,0], cls_segms_flat[:,1], cls_segms_flat[:,2])))
#     cluster_tuple = set(list(zip(clusters[:,0], clusters[:,1], clusters[:,2])))
#     intersect = segm_tuple.intersection(cluster_tuple)

#     cls_segms_flat = np.array(list(intersect))

#     color = colors[0]
#     trace += scatter_points(cls_segms_flat, markersize=1.5, opacity=0.1, color=color)
#     trace[-1].name = 'cluster %d' % 0

    # if not ground_truth:
    #     for idx, c in enumerate(np.unique(cls_segms_flat[:,3])):
    #         color   = colors[idx % len(colors)]
    #         cluster = cls_segms_flat[cls_segms_flat[:,3]==c]
    #         trace += scatter_points(cluster, markersize=1.5, color=color)
    #         trace[-1].name = 'cluster %d' % idx
    # else:
    #     for idx, c in enumerate(np.unique(clusters[:,5])):
    #         color   = colors[idx % len(colors)]
    #         cluster = clusters[clusters[:,5]==c]
    #         trace += scatter_points(cluster,markersize=1.5,color=color)
    #         trace[-1].name = 'cluster %d' % idx

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

def prep_pred_vis(bbox):

    for i in range(1, len(bbox)):
        cls_ids = np.full((bbox[i].shape[0], 1), i)
        bbox[i] = np.hstack((bbox[i], cls_ids))

    bbox = np.vstack((bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]))

    return bbox

def prep_gt_vis(input_data):

    boxes = input_data[:, :6]
    gt_class = input_data[:, 6] + 1
    gt_class = gt_class.reshape((gt_class.shape[0], 1))
    scores = np.ones((boxes.shape[0], 1)).astype(np.float32)

    return np.hstack((boxes, scores, gt_class))
