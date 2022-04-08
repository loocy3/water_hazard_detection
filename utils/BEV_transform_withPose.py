
import cv2
import numpy as np
from plane_fitting import fitPlane, getInliers, getHorizonLine, track3D_img, track3D_mask, fitPlaneCorr
from utilities import getStereoDisparity
# from dataloader import load_dataset_path
import os
import re
import zedutils
import torch.nn.functional as F
import torch
from torchvision import transforms
from scipy.io import loadmat
from PIL import Image

With_cur_r = False
BEV_flag = 0# 0: no bev, 1: my bev. 2:ori bev
VISO = True #True
debug = False
sequence_len = 16

ToTensor = transforms.Compose([
    transforms.ToTensor()])

root_dir = '../../../Dataset/'
save_dir = 'viso'
sequence = 8


map1x, map1y, map2x, map2y, mat, Q1 = \
                zedutils.getTransformFromConfig('SN1994.conf', Type='CAM_HD')
K_l, K_r, R_lr, T_lr= zedutils.getKRTInfo('SN1994.conf', Type='CAM_HD')
bev_size = 720
ori_size = (720, 1280)

Q = np.array([[1, 0, 0, -6.57925000e+02],
             [0, 1, 0, -3.78654000e+02],
             [0, 0, 0, 6.99943000e+02],
             [0, 0, 1/120., 0]])

def load_dataset_path(txt_path):
    p = np.genfromtxt(txt_path, dtype='str')
    return p[:, 1]

def cam_to_bev(cam_img, camera_k, bev_size, RT=None):
    # inputs:
    #   cam_img: ground image:
    #   camera_k: 3*3 K matrix of left color camera : 3*3
    #   bev_size: H=W
    #   RT: rotation and translate between cameras
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
    center = torch.tensor([bev_size//2-1,bev_size-1])
    uv = uv-center

    meter_per_pixel = 40/bev_size # can see 40 meter in the bev window
    bev2cam = meter_per_pixel * torch.tensor([[1, 0], [0, -1]]).float()  # shape = [2,2] # x = bev_u, z = -bev_v

    # Trans matrix from sat to realword
    XZ = torch.einsum('ij, hwj -> hwi', bev2cam, uv)
    Y = torch.ones((bev_size, bev_size, 1))*camera_height
    XYZ = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:]], dim=-1)  # [H,W,3]

    #default R
    # Rx, _ = cv2.Rodrigues(np.array([0.015, 0, 0])) #(np.array([0.027, 0, 0])) # use different angle for off-road 0.030 and on-road 0.015
    # if RT is not None:
    #     RT = Rx@RT
    # else:
    #     RT = np.hstack((Rx, torch.zeros((3, 1))))
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

def save_image(image, save_dir, sub_dir, file_num, file_id):
    dir = os.path.join(save_dir, sub_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    rgb_file_dir = os.path.join(dir, 'img_%09d' % (file_num))
    if not os.path.exists(rgb_file_dir ):
        os.makedirs(rgb_file_dir )

    rgb_file = os.path.join(rgb_file_dir, 'f_%09d.png' % (file_id))
    #cv2.imwrite(rgb_file,image)
    image.save(rgb_file)

    return

if __name__ == '__main__':
    save_dir = os.path.join(root_dir, save_dir)

    # read motion
    motion_file_on = os.path.join(save_dir, "video_on_road_motion.mat")
    motion_dict_on = loadmat(motion_file_on)
    motion_file_off = os.path.join(save_dir, "video_off_road_motion.mat")
    motion_dict_off = loadmat(motion_file_off)

    # read r
    if With_cur_r:
        r_file_on = os.path.join(save_dir, "video_on_road_r.mat")
        r_dict_on = loadmat(r_file_on)
        r_file_off = os.path.join(save_dir, "video_off_road_r.mat")
        r_dict_off = loadmat(r_file_off)

    camera_k = mat
    bev_size = 720
    modes = ['train', 'test']
    for mode in modes:
        txt_path = os.path.join(root_dir, 'both_road_' + mode + '.txt')
        mask_file_names = load_dataset_path(txt_path)
        for mask_file_name in mask_file_names:
            # test on raod
            #mask_file_name = '../Dataset/masks/on_road/left_mask_0000001035.png' # 0.022~27
            #mask_file_name = '../Dataset/masks/on_road/left_mask_0000003060.png' # 0.022

            #mask_file_name = '../Dataset/masks/off_road/left_mask_000000267.png' # 0.032
            #mask_file_name = '../Dataset/masks/off_road/left_mask_000005551.png' #0.034

            file_num = int(re.findall('\d+', mask_file_name)[0])
            if 'on_road' in mask_file_name:
                fname = 'masks/on_road/left_mask_%09d.png' % (file_num)
                sub_dir = 'video_on_road'
                motion_dict = motion_dict_on
                if With_cur_r:
                    r_dict = r_dict_on

                #continue # for different rx
            else:
                fname = 'masks/off_road/left_mask_%09d.png' % (file_num)
                sub_dir = 'video_off_road'
                motion_dict = motion_dict_off
                if With_cur_r:
                    r_dict = r_dict_off

                #continue  # for different rx

            # process mask
            dir = os.path.join(save_dir, sub_dir)
            if not os.path.exists(dir):
                os.makedirs(dir)
            mask_name = os.path.join(dir, 'left_mask_%09d.png' % (file_num))

            # if os.path.exists(mask_name):
            #     continue

            fname = os.path.join(root_dir, fname)
            mask = cv2.imread(fname, 0)  # grey sclae

            # mask rectify # need??
            mask = cv2.remap(mask, map1x, map1y,
                                   cv2.INTER_CUBIC)

            if With_cur_r and 'img_%09d' % (file_num) in r_dict.keys():
                R = r_dict['img_%09d' % (file_num)]
            else:
                R = np.eye(3)

            # test
            #R,_ = cv2.Rodrigues(np.array([0.034, 0, 0.00298545])) #RX_HD=0.00555597 RZ_HD=-0.00298545

            RT = np.hstack((R, np.zeros((3, 1))))
            bev_mask = cam_to_bev(mask, camera_k, bev_size, RT=RT)

            # save mask
            bev_mask.save(mask_name)

            # process sequence of rgb
            motion = np.eye(4)
            for file_id in range(file_num,file_num-sequence_len,-1):
                rgb_file_name = os.path.join(root_dir, sub_dir, 'img_%09d.ppm'%(file_id))
                pair = cv2.imread(rgb_file_name)
                if pair is None:
                    continue
                frame_left = pair[:, :pair.shape[1] // 2, :].copy()
                frame_right = pair[:, pair.shape[1] // 2:, :].copy()

                # image rectify # need??
                frame_left = cv2.remap(frame_left, map1x, map1y,
                                       cv2.INTER_CUBIC)
                frame_right = cv2.remap(frame_right, map2x, map2y,
                                        cv2.INTER_CUBIC)

                # turn camera image to bev image
                RT_motion = RT[:3, :3] @ motion[:3]
                bev_left = cam_to_bev(frame_left, camera_k, bev_size, RT=RT_motion)
                bev_right = cam_to_bev(frame_right, camera_k, bev_size, RT=RT_motion)

                if 0:
                    bev_left.save('left_%d.png' % (file_id))

                # debug merge left and mask, right and mask
                if 0:
                    if file_id == file_num:
                        bev_left.save('left_%d.png' % (file_id))
                        bev_right.save('right_%d.png' % (file_id))
                        bev_mask_a = bev_mask.convert("RGBA")
                        bev_left_a = bev_left.convert("RGBA")
                        bev_right_a = bev_right.convert("RGBA")
                        merge_lm = Image.blend(bev_left_a, bev_mask_a, alpha=.5)
                        merge_lm.save('merge_lm.png')
                        merge_rm = Image.blend(bev_right_a, bev_mask_a, alpha=.5)
                        merge_rm.save('merge_rm.png')

                # debug merge left and right
                if 0:
                    bev_left_a = bev_left.convert("RGBA")
                    bev_right_a = bev_right.convert("RGBA")
                    merge_lr = Image.blend(bev_left_a, bev_right_a, alpha=.5)
                    merge_lr.save('merge_lr.png')

                # debug merge left sequence
                if 0:
                    if file_id == file_num:
                        merge_seq = bev_left.convert("RGBA")
                    else:
                        bev_left_a = bev_left.convert("RGBA")
                        merge_seq = Image.blend(merge_seq, bev_left_a, alpha=.5)
                        merge_seq.save('merge_seq_%d.png'%(file_id))

                # motion update
                if 'img_%09d'%(file_id) in motion_dict.keys():
                    motion_update = motion_dict['img_%09d'%(file_id)]

                    R = motion[:3,:3] @ np.linalg.inv(motion_update[:3,:3])
                    t = motion[:3,-1:] - (motion[:3,:3] @ motion_update[:3,-1:])
                    motion = np.vstack(( np.hstack((R,t)), motion[-1:]))

                # save rgb image
                rgb_dir = os.path.join(save_dir, sub_dir)
                save_image(bev_left, rgb_dir, 'left', file_num, file_id)
                save_image(bev_right, rgb_dir, 'right', file_num, file_id)

    print("bev trans finish")
