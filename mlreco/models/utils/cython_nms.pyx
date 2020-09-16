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
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b) nogil:
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b) nogil:
    return a if a <= b else b

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def nms(np.ndarray[np.float32_t, ndim=2] dets, np.float32_t thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] z1 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 4]
    cdef np.ndarray[np.float32_t, ndim=1] z2 = dets[:, 5]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 6]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j, _k
    # sorted indices
    cdef int i, j, k
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, iz1, ix2, iy2, iz2, iarea
    cdef np.float32_t jx1, jy1, jz1, jx2, jy2, jz2, jarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, zz1, xx2, yy2, zz2
    cdef np.float32_t w, h, d
    cdef np.float32_t inter, ovr

    with nogil:
      for _i in range(ndets):
          i = order[_i]
          if suppressed[i] == 1:
              continue
          ix1 = x1[i]
          iy1 = y1[i]
          iz1 = z1[i]
          ix2 = x2[i]
          iy2 = y2[i]
          iz2 = z2[i]
          iarea = areas[i]
          for _j in range(_i + 1, ndets):
              j = order[_j]
              if suppressed[j] == 1:
                  continue
              jx1 = x1[j]
              jy1 = y1[j]
              jz1 = z1[j]
              jx2 = x2[j]
              jy2 = y2[j]
              jz2 = z2[j]
              jarea = areas[j]
              for _k in range(_j + 1, ndets):
                  k = order[_k]
                  xx1_t = max(ix1, jx1)
                  xx1 = max(xx1_t, x1[k])
                  yy1_t = max(iy1, jy1)
                  yy1 = max(yy1_t, y1[k])
                  zz1_t = max(iz1, jz1)
                  zz1 = max(zz1_t, z1[k])
                  xx2_t = min(ix2, jx2)
                  xx2 = min(xx2_t, x2[k])
                  yy2_t = min(iy2, jy2)
                  yy2 = min(yy2_t, y2[k])
                  zz2_t = min(iz2, jz2)
                  zz2 = min(zz2_t, z2[k])
                  w = max(0.0, xx2 - xx1 + 1)
                  h = max(0.0, yy2 - yy1 + 1)
                  d = max(0.0, zz2 - zz1 + 1)
                  inter = w * h * d
                  ovr = inter / (iarea + jarea + areas[k] - inter)
                  if ovr >= thresh:
                      suppressed[j] = 1

    return np.where(suppressed == 0)[0]

# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def soft_nms(
    np.ndarray[float, ndim=2] boxes_in,
    float sigma=0.5,
    float Nt=0.3,
    float threshold=0.001,
    unsigned int method=0
):
    boxes = boxes_in.copy()
    cdef unsigned int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1, x2, y1, y2, tx1, tx2, ty1, ty2, ts, area, weight, ov
    inds = np.arange(N)

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]
        ti = inds[i]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]
        inds[i] = inds[maxpos]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts
        inds[maxpos] = ti

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below
        # threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]

                    # if box score falls below threshold, discard the box by
                    # swapping with last box update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        inds[pos] = inds[N-1]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    return boxes[:N], inds[:N]
