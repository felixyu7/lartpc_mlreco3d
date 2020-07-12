# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    import cv2
except:
    pass
import numpy as np
import os
try:
    import pycocotools.mask as mask_util
except:
    pass

from utils.colormap import colormap
import utils.keypoints as keypoint_utils

# Use a non-interactive backend
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator
except:
    pass


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def vis_bbox_opencv(img, bbox, thick=1):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf', plain_img=False, no_adc=False, show_roi_num=False, entry=-1,
        run=-1, subrun=-1,event=-1):
    """Visual debugging of detections."""
    # print("SHAPE DESIRED:", boxes.shape)

    # for ll in range(len(boxes)):
    #     print(boxes[ll])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print("SHAPE: ",im.shape)
    if plain_img:
        #need something there
        boxes = np.array([[50,50,60,60,.99],[1,1,5,5,.99]])

    if no_adc:
        im.fill(0.0)

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)
    else:
        classes =[]
        for i in range(len(boxes)):
            classes.append(1)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return


    if segms is not None:
        masks = mask_util.decode(segms)


    color_list = colormap(rgb=True) / 255

    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)

    height, width, __ = im.shape
    im_grey = np.zeros ((height,width), np.float32)
    for dim1 in range(len(im)):
        for dim2 in range(len(im[0])):
            if im[dim1][dim2][0] > 250:
                value = 250
            elif im[dim1][dim2][0] < 0:
                value = 0
            else:
                value = im[dim1][dim2][0]
            im_grey[dim1][dim2] = value
    # np.set_printoptions(threshold=np.inf)
    # print(im_grey)
    ax.imshow(im_grey, interpolation='none',cmap='jet')

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    mask_color_id = 1
    once =1

    if plain_img:
        output_name = os.path.basename(im_name) + '.' + ext
        fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
        output_name = os.path.basename(im_name) + '.png'
        fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)

        plt.close('all')


    else:
        if entry != -1:
            ax.text(
                10,500,
                "Entry #"+str(entry),
                fontsize=12,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')
        if run != -1:
            ax.text(
                10,500,
                "Run: "+str(run) + " Subrun: " + str(subrun) + " Event: " + str(event),
                fontsize=12,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')
        for i in sorted_inds:
            # if classes[i] != 5:
            #     continue
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < thresh:
                continue

            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]

            print(dataset.classes[classes[i]], score)
            # show box (off by default, box_alpha=0.0)
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=False, edgecolor=color_mask,
                              linewidth=2.5, alpha=box_alpha))

            if show_class:
                # if (bbox[1] < )
                ax.text(
                    bbox[0], bbox[1] + 20,
                    get_class_string(classes[i], score, dataset),
                    fontsize=12,
                    family='serif',
                    bbox=dict(
                        facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                    color='white')
                # ax.text(
                #     bbox[0], bbox[1] +20,
                #     str(bbox[0])+" , "+ str(bbox[1]) + " , " + str(bbox[2])+" , "+ str(bbox[3]) ,
                #     fontsize=3,
                #     family='serif',
                #     bbox=dict(
                #         facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                #     color='white')

            if show_roi_num:
                dist=+10
                if show_class:
                    dist=20
                ax.text(
                    bbox[0], bbox[1] +dist,
                    str(i),
                    fontsize=12,
                    family='serif',
                    bbox=dict(
                        facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                    color='white')
                # ax.text(
                #     bbox[0], bbox[1] +dist+10,
                #     str(bbox[0])+" , "+ str(bbox[1]) + " , " + str(bbox[2])+" , "+ str(bbox[3]) ,
                #     fontsize=3,
                #     family='serif',
                #     bbox=dict(
                #         facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                #     color='white')

            # show mask
            # print('About to check the segms')
            if segms is not None and len(segms) > i and True:
                # print('Inside Segments')
                # for k,v in segms[0].items():
                #     print(k)
                # print("         ",i)
                # print('             Size: ', segms[i]['size'])
                # print()
                # print('             Count ', type(segms[i]['counts']) , segms[i]['counts'])
                # print()
                # print('             masks:', type(masks), masks.shape)


                if once:
                    np.set_printoptions(threshold=np.inf)
                    # print(i)
                    # print("Length masks[:,:,i]", len(masks[:,:,i]))
                    # print("Length masks", len(masks))
                    # print("Length masks[0]", len(masks[0]))
                    # print("Length masks[0][0]", len(masks[0][0]))
                    # print("Length masks[0][0][0]", len(masks[0][0][0]))




                    once =False
                e = masks[:, :, i]
                # print('             e:', type(e), e.shape, np.sum(e), np.amax(e))
                # print('             Box Dim:', bbox[2] - bbox[0] , bbox[3] - bbox[1] )

                _, contour, hier = cv2.findContours(
                    e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                # ax.imshow(e.copy())
                for c in contour:
                    polygon = Polygon(
                        c.reshape((-1, 2)),
                        fill=True, facecolor=color_mask,
                        edgecolor=color_mask, linewidth=.4,
                        alpha=0.5)
                    ax.add_patch(polygon)

            # # show keypoints
            # if keypoints is not None and len(keypoints) > i:
            #     kps = keypoints[i]
            #     plt.autoscale(False)
            #     for l in range(len(kp_lines)):
            #         i1 = kp_lines[l][0]
            #         i2 = kp_lines[l][1]
            #         if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            #             x = [kps[0, i1], kps[0, i2]]
            #             y = [kps[1, i1], kps[1, i2]]
            #             line = ax.plot(x, y)
            #             plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
            #         if kps[2, i1] > kp_thresh:
            #             ax.plot(
            #                 kps[0, i1], kps[1, i1], '.', color=colors[l],
            #                 markersize=3.0, alpha=0.7)
            #         if kps[2, i2] > kp_thresh:
            #             ax.plot(
            #                 kps[0, i2], kps[1, i2], '.', color=colors[l],
            #                 markersize=3.0, alpha=0.7)
            #
            #     # add mid shoulder / mid hip for better visualization
            #     mid_shoulder = (
            #         kps[:2, dataset_keypoints.index('right_shoulder')] +
            #         kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            #     sc_mid_shoulder = np.minimum(
            #         kps[2, dataset_keypoints.index('right_shoulder')],
            #         kps[2, dataset_keypoints.index('left_shoulder')])
            #     mid_hip = (
            #         kps[:2, dataset_keypoints.index('right_hip')] +
            #         kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            #     sc_mid_hip = np.minimum(
            #         kps[2, dataset_keypoints.index('right_hip')],
            #         kps[2, dataset_keypoints.index('left_hip')])
            #     if (sc_mid_shoulder > kp_thresh and
            #             kps[2, dataset_keypoints.index('nose')] > kp_thresh):
            #         x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
            #         y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
            #         line = ax.plot(x, y)
            #         plt.setp(
            #             line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            #     if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            #         x = [mid_shoulder[0], mid_hip[0]]
            #         y = [mid_shoulder[1], mid_hip[1]]
            #         line = ax.plot(x, y)
            #         plt.setp(
            #             line, color=colors[len(kp_lines) + 1], linewidth=1.0,
            #             alpha=0.7)

            output_name = os.path.basename(im_name) + '.' + ext
            fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
            output_name = os.path.basename(im_name) + '.png'
            fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
            plt.close('all')
