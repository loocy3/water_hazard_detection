# Chuong Nguyen 2016
# Water detection project
#
# Helper functions for warping images
# Author: Chuong Nguyen <chuong.nguyen@anu.edu.au>
#
# License: BSD 3 clause

import cv2
import numpy as np


def warpStereo(image, disp, mask=None):
    # warp image where disparity get
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    xx, yy = np.meshgrid(x, y)
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    if mask is None:
        xx += disp
    else:
        xx[mask] += disp[mask]

    imageWarped = cv2.remap(image, xx, yy, cv2.INTER_CUBIC)
    return imageWarped


def warp3D(image0, disp0, plane_coefs0, Q, mat, rvec, tvec, d_thres=None, val=0,
           sigma=None, useFittedPlane=True):
    ''' Warp image by reproject image to ground plane and project it to
    image space of the next frame.

    Input:
    - image0: image to warp
    - disp0: stereo disparity in reference of image0
    - plane_coefs0: plane coefficients of ground plane disparity
    - Q: reprojection matrix to convert disparity to 3D coordinates
    - mat: camera matrix
    - rvec, tvec: rotation and translation vectors
    - d_thres, val: disparity threshold below which the image is set to val
    - sigma: STD of Gaussian distribution for image blurring

    Output:
    - image0_warp: warped image
    '''
    u, v = np.meshgrid(np.arange(image0.shape[1]), np.arange(image0.shape[0]))
    d = plane_coefs0[0]*u + plane_coefs0[1]*v + plane_coefs0[2]

    if d_thres is not None:
        image0[d < d_thres] = val

    dd = d.copy()
    dd[d < 0] = 0
    if useFittedPlane:
        xyz = cv2.reprojectImageTo3D(dd.astype(np.float32), Q)
    else:
        xyz = cv2.reprojectImageTo3D(disp0.astype(np.float32), Q)
    xyz3 = xyz.reshape([xyz.shape[0]*xyz.shape[1], xyz.shape[2]])
    useMotion = True
    if useMotion:
        uv2, _ = cv2.projectPoints(xyz3, rvec, tvec, mat,
                                   np.array([0.0, 0.0, 0.0, 0.0]))
    else:
        uv2, _ = cv2.projectPoints(xyz3, rvec*0, tvec*0, mat,
                                   np.array([0.0, 0.0, 0.0, 0.0]))
    uv = uv2.reshape([image0.shape[0], image0.shape[1], -1])
    if sigma is None:
        image0_warped = cv2.remap(image0, uv[:,:,0].astype(np.float32),
                                  uv[:,:,1].astype(np.float32), cv2.INTER_LINEAR,
                                    borderValue=val)
    else:
        image0_blurred = cv2.blur(image0, (sigma, sigma))
        image0_warped = cv2.remap(image0_blurred, uv[:,:,0].astype(np.float32),
                                  uv[:,:,1].astype(np.float32), cv2.INTER_LINEAR,
                                  borderValue=val)
    # make sure probability above horizon stay the same
    image0_warped[d < 0] = image0[d < 0]
    return image0_warped
