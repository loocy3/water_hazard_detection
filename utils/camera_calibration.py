
import cv2
import numpy as np
import math
import zedutils
import os

root = '/home/shan/Dataset/water_hazard/'
bev_folder = 'viso'
cali_off_road = 503 #'video_off_road/ .ppm
cali_on_road = 2611 #video_on_road/

K_l, K_r, R_lr, T_lr= zedutils.getKRTInfo('SN1994.conf', Type='CAM_HD')

def get_off_hmatrix():
    #cali_off_road = 500 #'video_off_road/ .ppm
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

def get_on_hmatrix():
    #cali_on_road = 2611 #video_on_road/
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

def angle_from_Rmatrix(R):
    '''

    :param R:
    :return: roll x, pitch:y, yaw:z
    '''
    if R[2,0] != 1 and R[2,0] != -1:
         pitch_1 = -1*math.asin(R[2,0])
         pitch_2 = math.pi - pitch_1
         roll_1 = math.atan2( R[2,1] / math.cos(pitch_1) , R[2,2] /math.cos(pitch_1) )
         roll_2 = math.atan2( R[2,1] / math.cos(pitch_2) , R[2,2] /math.cos(pitch_2) )
         yaw_1 = math.atan2( R[1,0] / math.cos(pitch_1) , R[0,0] / math.cos(pitch_1) )
         yaw_2 = math.atan2( R[1,0] / math.cos(pitch_2) , R[0,0] / math.cos(pitch_2) )

         # IMPORTANT NOTE here, there is more than one solution but we choose the first for this case for simplicity !
         # You can insert your own domain logic here on how to handle both solutions appropriately (see the reference publication link for more info).
         sol_1 = np.array([roll_1, pitch_1, yaw_1])
         sol_2 = np.array([roll_2, pitch_2, yaw_2])
    else:
         yaw = 0 # anything (we default this to zero)
         if R[2,0] == -1:
            pitch = pi/2
            roll = yaw + math.atan2(R[0,1],R[0,2])
         else:
            pitch = -pi/2
            roll = -1*yaw + math.atan2(-1*R[0,1],-1*R[0,2])
         sol_1 = np.array([roll, pitch, yaw])
         sol_2 = None

    # convert from radians to degrees
    if sol_1 is not None:
        sol_1 = sol_1*180./math.pi
    if sol_2 is not None:
        sol_2 = sol_2*180/math.pi

    return sol_1, sol_2

if __name__ == '__main__':
    on_ori2bev_warp = get_on_hmatrix()
    off_ori2bev_warp = get_off_hmatrix()
    for ori2bev_warp in (off_ori2bev_warp,on_ori2bev_warp):
        # get r from matrix
        motions = cv2.decomposeHomographyMat(ori2bev_warp, K_l)
        for i in range(motions[0]):
            R = motions[1][i]
            sol_1, sol_2 = angle_from_Rmatrix(R)
            print(i, sol_1, sol_2)

