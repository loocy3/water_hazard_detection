
import cv2
import numpy as np
import pyviso2.src.viso2 as viso2
from plane_fitting import fitPlane, getInliers, getHorizonLine, track3D_img, track3D_mask, fitPlaneCorr
from utilities import getStereoDisparity
from dataloader import load_dataset_path
import os
import re
import zedutils
import torch.nn.functional as F
import torch
from torchvision import transforms
from skimage.io import imread
from mayavi import mlab

BEV_flag = 0# 0: no bev, 1: my bev. 2:ori bev
VISO = True #True
debug = False
sequence_len = 32

ToTensor = transforms.Compose([
    transforms.ToTensor()])

root_dir = '../../../Dataset/'
if VISO:
    save_dir = 'viso'
    sequence = 8
elif BEV_flag:
    save_dir = 'bev'


map1x, map1y, map2x, map2y, mat, Q1 = \
                zedutils.getTransformFromConfig('SN1994.conf', Type='CAM_HD')
K_l, K_r, R_lr, T_lr= zedutils.getKRTInfo('SN1994.conf', Type='CAM_HD')
bev_size = 720

Q = np.array([[1, 0, 0, -6.57925000e+02],
             [0, 1, 0, -3.78654000e+02],
             [0, 0, 0, 6.99943000e+02],
             [0, 0, 1/120., 0]])

def bird_eye(image, warpMatrix):
    image = image[300:720, :] # crop ground part, need change !!!!!!!!!!!
    sz = image.shape
    warpped_img = cv2.warpPerspective(image, warpMatrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return warpped_img


def get_bird_eye(leftBGR, rightBGR, mask_gt=None ):
    # calculate disp map
    pair = np.concatenate((leftBGR, rightBGR), axis=1)
    maxDisparity = 32
    disp = getStereoDisparity(pair, dispRange=[0, maxDisparity])

    # find pixel in a triangle ,in front of camera
    u, v = np.meshgrid(np.arange(disp.shape[1]), np.arange(disp.shape[0]))
    # mask = u/10 v/10, in a triangle, in front of camera.  groundCoefs: normal image width, height and disparity scale
    groundCoefs, mask = fitPlane(u, v, disp, skip=10, infHeight=350)
    d = groundCoefs[0] * u + groundCoefs[1] * v + groundCoefs[2]
    d[d < 0] = 0
    mask = getInliers(u, v, d, disp, tolerance=np.array([1, 0.05]))
    mask[:400, :] = False

    mask2 = np.logical_not(mask)
    disp2 = np.copy(disp)
    disp2[mask2] = 0 # disp with out liners
    disp3 = np.copy(disp)
    disp3[mask] = 0 # disp with in liners

    infHeight = 400
    w, h, dmax = 1.0, 1.0, 1.0
    v0 = h * infHeight / d.shape[0]  #
    u0, v0, d0 = [w / 2.0, v0, 0.0]
    u1, v1, d1 = [w / 2.0, h, dmax]
    u2, v2, d2 = [0.0, v0, 0.0]
    a = -d1 * (v2 - v0) / u0 / (v1 - v0)
    b = d1 / (v1 - v0)
    c = d1 * (v2 - 2.0 * v0) / (v1 - v0)

    x0 = np.array([a, b, c])

    # find width, height and disparity scale
    groundCoefs2 = fitPlaneCorr([groundCoefs[0], groundCoefs[1], groundCoefs[2] / 1000], leftBGR, rightBGR, mask)
    groundCoefs2[2] *= 1000
    # print(groundCoefs, groundCoefs2)
    ROIXZ = [-10., 10., 0.5, 20.]

    # BEV image, left image to groundCoefs2 scale
    left_img = track3D_img(leftBGR, groundCoefs2, disp, Q, ROIXZ)
    if mask_gt !=None:
        mask = track3D_mask(mask_gt[:, :, 0], groundCoefs2, disp, Q, ROIXZ)

        if debug:
            cv2.imwrite('mask_bev.png', mask)
    else:
        mask = None

    if debug:
        cv2.imwrite('left_bev.png', left_img)
    return left_img, mask


def get_hmatrix():
    origin1 = [491, 714]
    origin2 = [1251, 717]
    origin3 = [554, 545]
    origin4 = [873, 512]
    corner_points_array = np.float32([origin1, origin2, origin3, origin4])
    birdeye1 = [585, 620]
    birdeye2 = [810, 625]
    birdeye3 = [578, 514]
    birdeye4 = [778, 485]
    img_params = np.float32([birdeye1, birdeye2, birdeye3, birdeye4])
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    return matrix


def getECCTransformation(src_img, tgt_img, warp_matrix):
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)
    if warp_matrix is None:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    warp_mode = cv2.MOTION_HOMOGRAPHY
    number_of_iterations = 10000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC(src_img, tgt_img, warp_matrix, warp_mode, criteria)
    return warp_matrix

def cam_to_bev_2(leftBGR, rightBGR, mask, warp_matrix):
    #warpMatrix = get_hmatrix()
    #warped_img = cv2.warpPerspective(cam_img, warpMatrix, tgt_size)  # WH
    warped_left, _ = get_bird_eye(leftBGR, rightBGR)
    if debug:
        cv2.imwrite('mid_warped_left.png',warped_left)
    _, l2bevMatrix = getECCTransformation(leftBGR, warped_left, warp_matrix)
    warped_left = bird_eye(leftBGR, l2bevMatrix)
    mask = bird_eye(mask, l2bevMatrix)
    warped_right = bird_eye(rightBGR, l2bevMatrix)
    return warped_left, mask, warped_right


def cam_to_bev(cam_img, camera_k, bev_size, RT=None):
    # inputs:
    #   cam_img: ground image:
    #   camera_k: 3*3 K matrix of left color camera : 3*3
    #   bev_size: H=W
    #   T: translate between cameras
    # return:
    #   bev_img: bird eye's view image with bev_size

    # turn np to torch
    cam_img = ToTensor(cam_img)
    camera_k = torch.from_numpy(camera_k).float()

    C, H, W = cam_img.shape
    camera_height = 1.77

    # get back warp matrix
    # meshgrid the bev pannel
    i = j = torch.arange(0, bev_size)
    ii,jj = torch.meshgrid(i, j) # i:h,j:w
    uv = torch.stack([jj, ii], dim=-1).float()  # shape = [bev_size, bev_size, 2]
    center = torch.tensor([bev_size//2-1,bev_size//2-1])
    uv = uv-center

    meter_per_pixel = 25/bev_size*2 # can see 25 meter in the bev window
    bev2cam = meter_per_pixel * torch.tensor([[1, 0], [0, -1]]).float()  # shape = [2,2] # x = bev_u, z = -bev_v

    # Trans matrix from sat to realword
    XZ = torch.einsum('ij, hwj -> hwi', bev2cam, uv)
    Y = torch.ones((bev_size, bev_size, 1))*camera_height
    XYZ = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:]], dim=-1)  # [H,W,3]

    # add RT
    if RT is not None:
        RT = torch.from_numpy(RT).float()
        ones = torch.ones((bev_size, bev_size, 1))
        XYZ1 = torch.cat([XYZ, ones], dim=-1)  # [H,W,4]
        XYZ = torch.einsum('ij, hwj -> hwi', RT, XYZ1)  # [H,W,3]

    # project to camera
    uv1 = torch.einsum('ij, hwj -> hwi', camera_k, XYZ) # shape = [H,W,3]
    # only need view in front of camera ,Epsilon = 1e-6
    uv_last = torch.maximum(uv1[:, :, 2:], torch.ones_like(uv1[:, :, 2:]) * 1e-6)
    uv = uv1[:, :, :2] / uv_last  # shape = [H, W,2]

    # lefttop to center
    uv_center = uv - torch.tensor([W // 2, H // 2])  # shape = [H, W,2]

    # u:south, v: up from center to -1,-1 top left, 1,1 buttom right
    scale = torch.tensor([W // 2, H // 2])
    uv_center /= scale

    bev_img = F.grid_sample(cam_img.unsqueeze(0), uv_center.unsqueeze(0), mode='bilinear',
                              padding_mode='zeros')

    if C == 1:
        mode = 'L'
    else:
        mode = 'RGB'
    bev_img = transforms.functional.to_pil_image(bev_img.squeeze(0),mode=mode)
    return bev_img

def save_bev_images(frame_left, frame_right, mask, save_dir, img_folder, file_num):
    if 0:
        warped_left, _ = get_bird_eye(frame_left, frame_right, None)
        cv2.imwrite('warped_left_old.png',warped_left)

    #R = np.linalg.inv(R)
    Rx, _ = cv2.Rodrigues(np.array([0.027, 0, 0]))
    RT = np.hstack((Rx, torch.zeros((3,1))))
    if frame_left is not None:
        frame_left = cam_to_bev(frame_left, K_l, bev_size, RT)
        left_rgb_dir = os.path.join(save_dir, img_folder, 'left')
        if not os.path.exists(left_rgb_dir):
            os.makedirs(left_rgb_dir)
        left_rgb_file = os.path.join(left_rgb_dir, 'img_%09d.png' % file_num)
        frame_left.save(left_rgb_file)

    if mask is not None:
        mask = cam_to_bev(mask, K_l, bev_size, None)#RT)
        mask_dir = os.path.join(save_dir, 'masks', img_folder[6:])
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        mask_file = os.path.join(mask_dir, 'left_mask_%09d.png' % file_num)
        mask.save(mask_file)

    if frame_right is not None:
        RT = np.hstack((Rx@R,np.expand_dims(T/1000, axis=1)))
        frame_right = cam_to_bev(frame_right, K_r, bev_size, RT)
        right_rgb_dir = os.path.join(save_dir, img_folder, 'right')
        if not os.path.exists(right_rgb_dir):
            os.makedirs(right_rgb_dir)
        right_rgb_file = os.path.join(right_rgb_dir, 'img_%09d.png' % file_num)
        frame_right.save(right_rgb_file)

def viso_init():
    # set the most relevant parameters
    params = viso2.Stereo_parameters()
    params.calib.f = mat[0,0]
    params.calib.cu = mat[0,2]
    params.calib.cv = mat[1,2]
    params.base = -T_lr[0]/1000

    # initialize visual odometry
    viso = viso2.VisualOdometryStereo(params)
    recon = viso2.Reconstruction()
    recon.setCalibration(params.calib.f, params.calib.cu, params.calib.cv)

    return viso, recon

def tran_shift(image, K, pose, image_target):
    # h,w,c = image.shape
    # # meshgrid the right
    # i = torch.arange(0, h)
    # j = torch.arange(0, w)
    # ii, jj = torch.meshgrid(i, j)  # vi:h,uj:w
    # one = torch.ones_like(ii)
    # # uv1 = torch.stack([jj, ii, one], dim=-1).float()  # [h, w, 3]
    # uv = torch.stack([jj, ii], dim=-1).float()  # shape = [h, w, 2]
    # center = torch.tensor([w//2 , h//2])
    # uv = uv - center
    # uv1 = torch.cat([uv, one[:,:,None]], dim=-1).float()
    #
    # # Trans matrix from sat to realword
    # XYZ = torch.einsum('ij, hwj -> hwi', torch.from_numpy(np.linalg.inv(K)).float(), uv1)  # [h, w, 3]
    # XYZ1 = torch.cat([XYZ, one[:,:,None]], dim=-1).float()
    # pose_numpy = np.zeros((4,4))
    # pose.toNumpy(pose_numpy)
    # pose_numpy[:-1,-1] = 0 # for debug
    # XYZ1 = torch.einsum('ij, hwj -> hwi', torch.from_numpy(pose_numpy).float(), XYZ1) # [h, w, 4]
    # uv1 = torch.einsum('ij, hwj -> hwi', torch.from_numpy(K).float(), XYZ1[:,:,:-1])  # [H,W,3]
    # uv_last = torch.maximum(uv1[:, :, 2:], torch.ones_like(uv1[:, :, 2:]) * 1e-6)
    # uv = uv1[:, :, :2] / uv_last  # shape = [H, W,2]
    #
    # # lefttop to center
    # uv_center = uv - torch.tensor([w // 2, h // 2])  # shape = [H, W,2]
    #
    # # u:south, v: up from center to -1,-1 top left, 1,1 buttom right
    # scale = torch.tensor([w // 2, h // 2])
    # uv_center /= scale
    #
    # if not torch.is_tensor(image):
    #     image = ToTensor(image)
    # tran_image = F.grid_sample(image.unsqueeze(0), uv_center.unsqueeze(0), mode='bilinear',
    #                             padding_mode='zeros')
    # if c == 1:
    #     mode = 'L'
    # else:
    #     mode = 'RGB'
    # tran_image = transforms.functional.to_pil_image(tran_image.squeeze(0), mode=mode)
    # if debug:
    #     tran_image.save('shift_left.png')

    pose_numpy = np.zeros((4, 4))
    pose.toNumpy(pose_numpy)
    R = pose_numpy[:3,:3]
    T = pose_numpy[:3,-1:]
    n = np.array([[0,0,1]])
    #H = K@(R)@np.linalg.inv(K)
    H = K @ (R - T @ n) @ np.linalg.inv(K)

    # H = H.astype('float32')
    # H_refine = getECCTransformation(image, image_target, None)
    # print("H", H, "H_refine", H_refine)

    warped_img = cv2.warpPerspective(image, H, (image.shape[1],image.shape[0]))  # Image warping
    if debug:
        cv2.imwrite('warped_left.png', warped_img)
    return warped_img


def tran_shift_bev(image, K, pose):
    if pose != None:
        pose_numpy = np.zeros((4, 4))
        pose.toNumpy(pose_numpy)
        bev_img = cam_to_bev(image, K, bev_size, RT=pose_numpy[:3])
    else:
        bev_img = cam_to_bev(image, K, bev_size, RT=None)
    return bev_img

def tran_viso_images(viso, recon, left, right):
    # viso
    # left: gray left image list: from current to previous(in and out)
    # right: gray  right image list: from current to previous(in and out)

    pose = viso2.Matrix_eye(4)

    # inverser order
    tran_lefts = []
    tran_rights = []
    #pose_list = []
    for i in range(len(left)):
        # turn grb to grey
        left_grey = cv2.cvtColor(left[i], cv2.COLOR_BGR2GRAY)
        right_grey = cv2.cvtColor(right[i], cv2.COLOR_BGR2GRAY)

        if debug:
            cv2.imwrite('grey_left.png', left_grey)
            cv2.imwrite('grey_right.png', right_grey)

        if viso.process_frame(left_grey, right_grey):
            motion = viso.getMotion()
            est_motion = viso2.Matrix_inv(motion)
            pose = pose * est_motion # motion is 4*4

            num_matches = viso.getNumberOfMatches()
            num_inliers = viso.getNumberOfInliers()
            print('Matches:', num_matches, "Inliers:", 100 * num_inliers / num_matches, '%, Current pose:', pose) # pose is 4*4 matrix
            matches = viso.getMatches()
            assert (matches.size() == num_matches)
            recon.update(matches, motion, 0)

            if 0:  # debug:
                tran_left = tran_shift(left[i], mat, pose, left[i-1])
                cv2.imwrite('debug_last_motion.png',tran_left)
                cv2.imwrite('debug_current.png', left[i-1])
        else:
            print('.... failed!')
            # use last pose?

        # # warp with pose
        # tran_left = tran_shift(left[i], mat, pose, left[0])
        # tran_lefts.append(tran_left)
        #
        # # right is already in left camera RTK, but with disparity
        # tran_right = tran_shift(right[i], mat, pose, left[0])
        # tran_rights.append(tran_right)

        # warp with pose and bev
        tran_left_bev = tran_shift_bev(left[i], mat, viso2.Matrix_inv(pose))
        tran_lefts.append(tran_left_bev)
        tran_right_bev = tran_shift_bev(right[i], mat, viso2.Matrix_inv(pose))
        tran_rights.append(tran_right_bev)

    return tran_lefts, tran_rights

def save_viso_images(viso, recon, frames_left, frames_right, save_dir, img_folder, file_num):
    tran_left, tran_right = tran_viso_images(viso, recon, frames_left, frames_right)
    left_rgb_dir = os.path.join(save_dir, img_folder, 'left')
    if not os.path.exists(left_rgb_dir):
        os.makedirs(left_rgb_dir)
    right_rgb_dir = os.path.join(save_dir, img_folder, 'right')
    if not os.path.exists(right_rgb_dir):
        os.makedirs(right_rgb_dir)

    left_rgb_file_dir = os.path.join(left_rgb_dir, 'img_%09d' % (file_num))
    if not os.path.exists(left_rgb_file_dir):
        os.makedirs(left_rgb_file_dir)
    right_rgb_file_dir = os.path.join(right_rgb_dir, 'img_%09d' % (file_num))
    if not os.path.exists(right_rgb_file_dir):
        os.makedirs(right_rgb_file_dir)

    for i in range(len(tran_left)):
        left_rgb_file = os.path.join(left_rgb_file_dir, 'f_%09d.png' % (file_num-i))
        #cv2.imwrite(left_rgb_file,tran_left[i])
        tran_left[i].save(left_rgb_file)
        right_rgb_file = os.path.join(right_rgb_file_dir, 'f_%09d.png' % (file_num-i))
        #cv2.imwrite(right_rgb_file, tran_right[i])
        tran_right[i].save(right_rgb_file)

    #debug
    if 1:
        # merge left sequence
        # merge left right

    return

if __name__ == '__main__':
    save_dir = os.path.join(root_dir, save_dir)
    if 0:#BEV_flag == 1 or VISO:
        # process mask
        modes = ['train', 'test']
        for mode in modes:
            txt_path = os.path.join(root_dir, 'both_road_' + mode + '.txt')
            mask_file_names = load_dataset_path(txt_path)
            for mask_file_name in mask_file_names:
                file_num = int(re.findall('\d+', mask_file_name)[0])
                if 'on_road' in mask_file_name:
                    fname = 'masks/on_road/left_mask_%09d.png' % (file_num)
                    sub_dir = 'video_on_road'
                else:
                    fname = 'masks/off_road/left_mask_%09d.png' % (file_num)
                    sub_dir = 'video_off_road'

                fname = os.path.join(root_dir, fname)
                mask = cv2.imread(fname, 0)  # grey sclae
                save_bev_images(None, None, mask, save_dir, sub_dir, file_num)

    # left and right without mask
    if VISO:
        viso, recon = viso_init()
    dirs = ['video_on_road', 'video_off_road']
    for dir in dirs:
        file_names = os.listdir(os.path.join(root_dir, dir))

        # order by number
        file_names = sorted(file_names)

        frames_left = []
        frames_right = []
        pose = viso2.Matrix_eye(4)
        for file_name in file_names:
            if 'ppm' not in file_name:
                continue
            #if os.path.exists(os.path.join(save_dir, dir, 'left', file_name.replace("ppm", "png"))):
            if os.path.exists(os.path.join(save_dir, dir, 'left', file_name.replace(".ppm", ""))):
                continue
            file_num = int(re.findall('\d+', file_name)[0])
            # file_num = 466#3874 #test
            full_file_name = os.path.join(root_dir, dir, file_name)
            pair = cv2.imread(full_file_name)
            frame_left = pair[:, :pair.shape[1] // 2, :].copy()
            frame_right = pair[:, pair.shape[1] // 2:, :].copy()
            if debug:
                cv2.imwrite('left_ori.png',frame_left)
                cv2.imwrite('right_ori.png', frame_right)

            # image rectify
            frame_left = cv2.remap(frame_left, map1x, map1y,
                                cv2.INTER_CUBIC)
            frame_right = cv2.remap(frame_right, map2x, map2y,
                                cv2.INTER_CUBIC)
            if debug:
                cv2.imwrite('left_rec.png', frame_left)
                cv2.imwrite('right_rec.png', frame_right)

            if VISO:
                # viso, recon = viso_init()
                # create queue for left
                frames_left.append(frame_left)
                if len(frames_left) > sequence_len:
                    frames_left.pop(0)
                frames_right.append(frame_right)
                if len(frames_right) > sequence_len:
                    frames_right.pop(0)
                if len(frames_left) >= sequence_len:
                    save_viso_images(viso, recon, frames_left[::-1], frames_right[::-1], save_dir, dir, file_num)
                    #save_viso_images(viso, recon, frames_left, frames_right, save_dir, dir, file_num)
            elif BEV_flag==1:
                save_bev_images(frame_left, frame_right, None, save_dir, dir, file_num)
            elif BEV_flag==2:
                cam_to_bev_2(frame_left, frame_right, None, None)
    print("bev trans finish")
