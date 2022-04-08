# Chuong Nguyen 2016
# Water detection project
#
# This file contains mislaneous utility functions
# Author: Chuong Nguyen <chuong.nguyen@anu.edu.au>
#
# License: BSD 3 clause

import cv2
import numpy as np

def getStereoDisparityPair(leftFrame, rightFrame, window_size=19,
                           dispRange=[0, 32], scale=1, algorithm='SGBM'):
    '''
    Compute stereo disparity using semi-global block matching.

    Input:
        leftFrame, rightFrame: left and right rectified images

    Output:
        disp: stereo disparity of the same size as input images

    '''
    if len(leftFrame.shape) > 2 or len(rightFrame.shape) > 2:
        leftFrameGray = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
        rightFrameGray = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
    else:
        leftFrameGray = leftFrame
        rightFrameGray = rightFrame

    num_disp = dispRange[1] - dispRange[0]
    if scale != 1:
        leftFrameGray = cv2.resize(leftFrameGray, None, fx=scale, fy=scale,
                                   interpolation = cv2.INTER_CUBIC)
        rightFrameGray = cv2.resize(rightFrameGray, None, fx=scale, fy=scale,
                                    interpolation = cv2.INTER_CUBIC)
        num_disp = int(dispRange[1]*scale - dispRange[0]*scale)

    if algorithm=='BM':
        # stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, ndisparities=num_disp,SADWindowSize=window_size)
        stereo = cv2.StereoBM_create(numDisparities=num_disp,
                                     blockSize=window_size)
        disp = stereo.compute(leftFrameGray, rightFrameGray)
    else: # algorithm='SBM'
        # stereo = cv2.StereoSGBM(dispRange[0], num_disp, window_size)
        stereo = cv2.StereoSGBM_create(dispRange[0], num_disp, window_size)
        disp = stereo.compute(leftFrameGray, rightFrameGray)

    # divide to 16 to get pixel displacement
    disp = (disp.astype(np.float32) + dispRange[0])/16.0
    if scale != 1:
        disp = cv2.resize(disp/scale, None, fx=1./scale, fy=1./scale,
                          interpolation = cv2.INTER_LINEAR)
    return disp


def getStereoDisparity(joinedFrame, window_size=19, dispRange=[0, 32], scale=1):
    '''
    Compute stereo disparity using semi-global block matching.

    Input:
        joinedFrame: concatenated left and right rectified images

    Output:
        disp: stereo disparity of the same size as input images

    '''
    frameGray = cv2.cvtColor(joinedFrame, cv2.COLOR_BGR2GRAY)
    leftFrame = frameGray[:, :frameGray.shape[1]//2]
    rightFrame = frameGray[:, frameGray.shape[1]//2:]
    return getStereoDisparityPair(leftFrame, rightFrame, window_size,
                                  dispRange, scale, algorithm='BM')
