#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import mlreco.models.sparse_resnet as sparse_resnet
import mlreco.models.RPN as RPN
import scipy
import mlreco.models.utils.boxes as box_utils
import mlreco.models.utils.blob as blob_utils
from mlreco.models.model.roi_layers import ROIPool, ROIAlign

import time


class MaskRCNN(nn.Module):
    
    def __init__(self, cfg, name="maskrcnn"):
        super(MaskRCNN,self).__init__()
        
        self.backbone = sparse_resnet.SparseResNet(cfg)
        self.Box_Head = sparse_resnet.ResNet_roi_conv5_head(512, self.roi_feature_transform, 1/16)
        self.RPN = RPN.RPN(cfg)
        
    def forward(self, input):
        return_dict = {}
        
        input_data = input[0]
        
        before_resnet = time.time()
        feature_map = self.backbone(input_data)
        print("Time taken in ResNet: ", time.time() - before_resnet)
        
        roidb = construct_roidb(input)
        
        before_construct = time.time()
        blobs, valid = get_minibatch(roidb)
        print("Time taken to construct roidb", time.time() - before_construct)
        
        before_rpn = time.time()
        rpn_ret = self.RPN(feature_map, torch.from_numpy(blobs['im_info']), blobs['roidb'])
        print("Time taken in RPN: ", time.time() - before_rpn)
        
        if self.training:
            box_head_ret, res5_feat = self.Box_Head(feature_map, rpn_ret)
        else:
            box_head_ret = self.Box_Head(feature_map, rpn_ret)
        
        
#         rpn_kwargs = {'rpn_labels_int32_wide': torch.Tensor(np.full(rpn_ret['rpn_cls_logits'].shape, -1)).cuda(),
#                      'rpn_bbox_targets_wide': torch.Tensor(np.zeros(rpn_ret['rpn_bbox_pred'].shape)).cuda(),
#                      'rpn_bbox_inside_weights_wide': torch.Tensor(np.zeros(rpn_ret['rpn_bbox_pred'].shape)).cuda(),
#                      'rpn_bbox_outside_weights_wide': torch.Tensor(np.zeros(rpn_ret['rpn_bbox_pred'].shape)).cuda()}
        
    
        
        
        loss_rpn_cls, loss_rpn_bbox = RPN.rpn_losses(rpn_ret, blobs)
        
        return_dict['loss_rpn_cls'] = loss_rpn_cls
        return_dict['loss_rpn_bbox'] = loss_rpn_bbox
        
        import pdb; pdb.set_trace()
    
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
        print(device_id)
        rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).to(torch.device(device_id))
        if method == 'RoIPoolF':
            xform_out = ROIPool((resolution, resolution), spatial_scale)(blobs_in, rois)
        elif method == 'RoIAlign':
            # print('RESOLUTION', resolution)
            xform_out = ROIAlign(
                (resolution, resolution, resolution), spatial_scale, sampling_ratio)(blobs_in, rois.float())

        return xform_out
    
        return return_dict

def construct_roidb(input):
    
    roidb = {}
    roidb['boxes'] = input[2][0].cpu().numpy()[:,:6].astype(float)
    roidb['segms'] = input[3][:,:3].cpu().numpy().astype(float)
    roidb['clusters'] = input[3].cpu().numpy().astype(float)
    roidb['gt_classes'] = (input[2][0].cpu().numpy()[:,6:].reshape(len(roidb['boxes'])) + 1).astype(int)
    

    gt_overlaps = np.zeros((len(roidb['boxes']), 6)).astype(float)
    box_to_gt_ind_map = np.zeros((len(roidb['boxes']))).astype(int)

    for x in range(roidb['gt_classes'].shape[0]):
        gt_overlaps[x][roidb['gt_classes'][x]] = 1

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
    
    im = np.zeros((1024, 1024, 1024, 3))
    
    for i in range(len(coords)):
        im[coords[i][0]][coords[i][1]][coords[i][2]][0]= values[i]
        im[coords[i][0]][coords[i][1]][coords[i][2]][1]= values[i]
        im[coords[i][0]][coords[i][1]][coords[i][2]][2]= values[i]
        
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
        im = prep_images(roidb[i]['clusters'])
#         im = cv2.imread(roidb[i]['image'])
#         assert im is not None, \
#             'Failed to read image \'{}\''.format(roidb[i]['image'])
#         # If NOT using opencv to read in images, uncomment following lines
#         # if len(im.shape) == 2:
#         #     im = im[:, :, np.newaxis]
#         #     im = np.concatenate((im, im, im), axis=2)
#         # # flip the channel, since the original one using cv2
#         # # rgb -> bgr
#         # im = im[:, :, ::-1]
#         if roidb[i]['flipped']:
#             im = im[:, ::-1, :]
        target_size = 1024
        im, im_scale = blob_utils.prep_im_for_blob(
            im, np.array([[[0.0, 0.0, 0.0]]]), [target_size], 1024)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]
    
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    blob_names += RPN.get_rpn_blob_names(is_training=is_training)

    return blob_names