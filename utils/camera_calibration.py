
import cv2
import numpy as np
import os
import re
import zedutils
import torch.nn.functional as F
import torch
from torchvision import transforms
from scipy.io import loadmat
from PIL import Image

cali_image = '../Dataset/video_off_road/img_000000500.ppm'

def get_hmatrix():
    origin1 = [491, 714]
    origin2 = [1251, 717]
    origin3 = [554, 545]
    origin4 = [873, 512]
    corner_points_array = np.float32([origin1,origin2,origin3,origin4])
    birdeye1 = [585,620]
    birdeye2 = [810,625]
    birdeye3 = [578,514]
    birdeye4 = [778,485]
    img_params = np.float32([birdeye1,birdeye2,birdeye3,birdeye4])
    matrix = cv2.getPerspectiveTransform(corner_points_array,img_params)
    return matrix

def getECCTransformation():
    pair = cv2.imread(cali_image)
    matrix = get_hmatrix()
    leftBGR = pair[:, :pair.shape[1]//2, :]
    rightBGR = pair[:, pair.shape[1]//2:, :]
    img_transformed_l = cv2.warpPerspective(leftBGR,matrix,(1280,720))
    img_transformed_r = cv2.warpPerspective(rightBGR,matrix,(1280,720))
    # img_transformed_l = cv2.warpPerspective(leftBGR,matrix,(647/2,189))
    # img_transformed_r = cv2.warpPerspective(rightBGR,matrix,(647/2,189))
    leftBGR_gray = cv2.cvtColor(img_transformed_l,cv2.COLOR_BGR2GRAY)
    rightBGR_gray  = cv2.cvtColor(img_transformed_r,cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    warp_mode = cv2.MOTION_HOMOGRAPHY
    number_of_iterations = 10000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC (rightBGR_gray,leftBGR_gray,warp_matrix, warp_mode, criteria)
    return warp_matrix