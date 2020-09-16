from __future__ import absolute_import
from __future__ import division

from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from functools import reduce
import operator

import mlreco.models.utils.net as net_utils
# import mlreco.models.upsample as mynn

#to Vis
import numpy as np
import time
try:
    import cv2
except:
    pass


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

class mask_rcnn_outputs(nn.Module):
    """Mask R-CNN specific outputs: either mask logits or probs."""
    def __init__(self, dim_in, validation=False):
        super(mask_rcnn_outputs,self).__init__()
        self.validation=validation
        self.dim_in = dim_in

        n_classes = 6
        # Predict mask using Conv
        self.classify = nn.Conv3d(dim_in, n_classes, 1, 1, 0)
#         self.upsample = mynn.BilinearInterpolation3d(
#             n_classes, n_classes, 16)
        self.upsample = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=False)
        self._init_weights()

    def _init_weights(self):

#         weight_init_func = MSRAFill
#         weight_init_func = partial(init.normal_, std=0.001)
        init.kaiming_normal_(self.classify.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.classify.bias, 0)

    def forward(self, x):
        x = self.classify(x)
        x = self.upsample(x)
        
        x = torch.sigmoid(x)
            
        return x


def mask_rcnn_losses(masks_pred, masks_int32, ):
    """Mask R-CNN specific losses."""
    # print('Taking a loss')
    # print('Shape of truth:', masks_int32.shape)
    # print('Shape of pred: ', masks_pred.shape)

    n_rois, n_classes, _, _, _ = masks_pred.size()
    device_id = ""
    if masks_pred.is_cuda:
        device_id = masks_pred.get_device()
    else:
        device_id = 'cpu'
    masks_gt = Variable(torch.from_numpy(masks_int32.astype('float32'))).to(torch.device(device_id))
    el1 = float(np.amax(masks_int32))
    el2 = float(torch.max(masks_gt))

#     weight = (masks_gt == 0).float() * 0.1 + (masks_gt > 0).float()
    # if el1 != el2:
        # print (el1,el2)

    #vis code
    
    weight_A = (masks_gt == 2)
    weight_B = (masks_gt == 1)
    
#     weight_C = (masks_gt == 0)

      # masks_int32 {1, 0, -1}, -1 means ignore
#     total_for_avg = weight.sum()
    
#     num_on  = torch.sum(masks_gt>0).item()
#     num_off = torch.sum(masks_gt==0).item()
#     num_inv = torch.sum(masks_gt==-1).item()
#     total_num = num_on+num_off
#     dim1, dim2 = masks_gt.shape
    # print('on: ', num_on, " off: ", num_off, " inv: ",num_inv)
    # print(dim1*dim2 , ", ", num_on+num_off+num_inv)
#     if num_off==0:
#         num_off=1
#     if num_on ==0:
#         num_on=1

#     weight = weight  * ( (1-masks_gt)*total_num/num_off + (masks_gt)*total_num/num_on )

#     resolution = cfg.MRCNN.RESOLUTION
    resolution = 192
    
    masks_gt[masks_gt == 1] = 0
    masks_gt[masks_gt == 2] = 1
    
    masks_gt_np = masks_gt.cpu().numpy()
    
#     masks_gt_np[masks_gt_np == 1] = 0
#     masks_gt_np[masks_gt_np == 2] = 1
    
    masks_pred_detach = masks_pred.detach().cpu()
    masks_pred_np = masks_pred_detach.numpy()
    masks_gt_np = np.reshape(masks_gt_np, (masks_gt_np.shape[0], 6, resolution, resolution, resolution))
    this_class = -1
    
#     for i in range(len(weight_A)):
#         print(i)
#         print('A: ', weight_A[i].sum())
#         print('B: ', weight_B[i].sum())
#         print()
    
#     masks_gt[masks_gt == 2] = 1
    
    acc_list = [[],[],[],[],[],[]]
    acc_acc = np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,0.,0.,0.])
    for roi in range(masks_gt_np.shape[0]):
        for clas in range(6):
            sum = masks_gt_np[roi][clas][:][:][:].sum()
            if (sum > 0):
                this_class = clas
#                 if this_class == 3:
#                     print('MUON: ', roi)               
                correct_on_pix = ((masks_gt_np[roi][clas]*masks_pred_np[roi][clas]) > 0.5)
                this_acc_num = correct_on_pix.sum()
                this_acc = float(this_acc_num)/float(sum)
                acc_list[clas].append(this_acc)

    full_acc_num = 0
    full_acc_den = 0
    for clas in range(6):
        full_acc_den = full_acc_den+len(acc_list[clas])
        if (len(acc_list[clas]) != 0):
            sum = 0
            for acc in acc_list[clas]:
                sum = sum + acc
                full_acc_num = full_acc_num + acc
            if (len(acc_list[clas]) != 0):
                acc_acc[clas] = float(sum)/float(len(acc_list[clas]))

    if (full_acc_den !=0):
        acc_acc[6] = float(full_acc_num)/float(full_acc_den)
    
    acc_acc = torch.from_numpy(acc_acc)
    
#     loss_A = F.binary_cross_entropy_with_logits(masks_pred.view(n_rois, -1), masks_gt, weight_A, reduction='sum') / weight_A.sum()
#     loss_B = F.binary_cross_entropy_with_logits(masks_pred.view(n_rois, -1), masks_gt, weight_B, reduction='sum') / weight_B.sum()
#     loss_C = F.binary_cross_entropy_with_logits(masks_pred.view(n_rois, -1), masks_gt, weight_C, reduction='sum') / weight_C.sum()

    loss_A = iou_pytorch(masks_pred.view(n_rois, -1), masks_gt, weight_A)
    loss_B = iou_B(masks_pred.view(n_rois, -1), masks_gt, weight_B)
#     loss_C = iou_pytorch(masks_pred.view(n_rois, -1), masks_gt, weight_C)
    loss = 0.4 * loss_A + 0.6 * loss_B
    print("loss_A: ", loss_A)
    print("loss_B: ", loss_B)
#     print("loss_C: ", loss_C)
    
    
#     loss = F.binary_cross_entropy_with_logits(masks_pred.view(n_rois, -1), masks_gt, weight, reduction='sum')

#     loss = 0.7 * loss_A + 0.3 * loss_B + 0.001 * loss_C
#     print("loss_A: ", loss_A)
#     print("loss_B: ", loss_B)
#     print("loss_C: ", loss_C)
    
    weight_A_np = np.reshape(weight_A.cpu().numpy(), (weight_A.cpu().numpy().shape[0], 6, resolution, resolution, resolution))
    weight_B_np = np.reshape(weight_B.cpu().numpy(), (weight_B.cpu().numpy().shape[0], 6, resolution, resolution, resolution))
#     weight_C_np = np.reshape(weight_C.cpu().numpy(), (weight_C.cpu().numpy().shape[0], 6, resolution, resolution, resolution))
    
    num = 0
    num_B = 0
    for roi in range(masks_gt_np.shape[0]):
        for clas in range(6):
            sum_A = (masks_gt_np[roi][clas][:][:][:] * weight_A_np[roi][clas]).sum()
            sum_B = (weight_B_np[roi][clas]).sum()
            if sum_A > 0:
                pred_pixels = ((masks_pred_np[roi][clas]) > 0.5)
                pred_zero = ((masks_pred_np[roi][clas]) <= 0.5) * (weight_A_np[roi][clas] + weight_B_np[roi][clas])
                pixels_A = pred_pixels * weight_A_np[roi][clas]
                
                pixels_A_off = pred_zero * weight_A_np[roi][clas]
                
                pixels_B = pred_pixels * weight_B_np[roi][clas]
                
#                 print("predictions on A: ", float(pixels_A.sum()) / float(sum_A))
                acc_acc[7] += float(pixels_A.sum()) / float(sum_A)
                if sum_B > 0:
#                     print("B pixels predicted as A: ", float(pixels_B.sum()) / float(sum_B))
                    acc_acc[8] += float(pixels_B.sum()) / float(sum_B)
                    num_B += 1
#                 print("A pixels predicted as B: ", float(pixels_A_off.sum()) / float(sum_A))
                acc_acc[9] += float(pixels_A_off.sum()) / float(sum_A)
                num += 1
                
    acc_acc[7] /= num
    acc_acc[8] /= num_B
    acc_acc[9] /= num
    
    b_pred = ((masks_pred.view(n_rois, -1) * weight_B) > 0.5)
    acc_acc[8] = b_pred.sum().float() / weight_B.sum().float()
    
    # print()
    # print('loss is type: ', type(loss))
    # print('loss shape is: ', loss.shape)
    # print()
#     loss /= total_for_avg

    return loss * 0.1, acc_acc

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, weight):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
#     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
#     outputs = outputs.detach()
#     outputs = (outputs > 0.5).int()
#     labels = labels.int()
    
#     on = torch.where(labels > -1)
#     on = (labels > -1)
    labels = labels * weight
    outputs = outputs * weight
    intersection = (outputs * labels).sum((1))
    union = ((outputs + labels) - (outputs * labels)).sum((1))
    
    for i in range(len(union)):
        if union[i] == 0:
            intersection[i] = 0
            union[i] = 1
    
    iou = (intersection) / (union)  # We smooth our devision to avoid 0/0
    print(iou)
    
#     if 1 - iou.mean() < 0:
#         import pdb; pdb.set_trace()
    
    return 1 - iou.mean()  # Or thresholded.mean() if you are interested in average across the batch

def iou_B(outputs, labels, weight):
    
    labels = labels * weight
    outputs = outputs * weight
    
    intersection = outputs.sum()
    union = weight.sum()
    
    return intersection.float() / union.float()

# ---------------------------------------------------------------------------- #
# Mask heads
# ---------------------------------------------------------------------------- #


class mask_rcnn_fcn_head_v0upshare(nn.Module):
    """Use a ResNet "conv5" / "stage5" head for mask prediction. Weights and
    computation are shared with the conv5 box head. Computation can only be
    shared during training, since inference is cascaded.

    v0upshare design: conv5, convT 2x2.
    """
    def __init__(self, dim_in, roi_xform_func, spatial_scale, validation=False):
        super(mask_rcnn_fcn_head_v0upshare,self).__init__()
        self.validation = validation
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = 256
        self.SHARE_RES5 = True

        self.res5 = None  # will be assigned later
        dim_conv5 = 2048
        self.upconv5 = nn.ConvTranspose3d(dim_conv5, self.dim_out, 2, 2, 0)
        
        self._init_weights()

    def _init_weights(self):
        MSRAFill(self.upconv5.weight)
        init.constant_(self.upconv5.bias, 0)

    def share_res5_module(self, res5_target):
        """ Share res5 block with box head on training """
        self.res5 = res5_target

    def forward(self, x, rpn_ret, roi_has_mask_int32=None):
        # print('Then I am here!')
        if self.training or self.validation:
            # On training, we share the res5 computation with bbox head, so it's necessary to
            # sample 'useful' batches from the input x (res5_2_sum). 'Useful' means that the
            # batch (roi) has corresponding mask groundtruth, namely having positive values in
            # roi_has_mask_int32.
            inds = np.nonzero(roi_has_mask_int32 > 0)[0]
            device_id = ''
            if x.is_cuda:
                device_id = x.get_device()
            else:
                device_id = 'cpu'

            inds = Variable(torch.from_numpy(inds)).to(torch.device(device_id))
            x = x[inds]
            # print("feat, upconv: ", x.shape)
        else:

            # On testing, the computation is not shared with bbox head. This time input `x`
            # is the output features from the backbone network
            x = self.roi_xform(
                x, rpn_ret,
                blob_rois='mask_rois',
                method='RoIAlign',
                resolution=12,
                spatial_scale=self.spatial_scale,
                sampling_ratio=-1
            )
            x = self.res5(x)
        x = self.upconv5(x)
        x = F.relu(x, inplace=True)
        return x

def MSRAFill(tensor):
    """Caffe2 MSRAFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_out = size / tensor.shape[1]
    scale = np.sqrt(2 / fan_out)
    return init.normal_(tensor, 0, scale)