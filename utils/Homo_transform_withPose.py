
import cv2
import numpy as np
from plane_fitting import fitPlane, getInliers, getHorizonLine, track3D_img, track3D_mask, fitPlaneCorr
from utilities import getStereoDisparity
from dataloader import load_dataset_path
import os
import re
import zedutils
import torch.nn.functional as F
import torch
from torchvision import transforms
from scipy.io import loadmat
from PIL import Image


ToTensor = transforms.Compose([
    transforms.ToTensor()])

root_dir = '../../../Dataset/'
save_dir = 'viso_homo'
sequence_len = 16


map1x, map1y, map2x, map2y, mat, Q1 = \
                zedutils.getTransformFromConfig('SN1994.conf', Type='CAM_HD')
K_l, K_r, R_lr, T_lr= zedutils.getKRTInfo('SN1994.conf', Type='CAM_HD')

Q = np.array([[1, 0, 0, -6.57925000e+02],
             [0, 1, 0, -3.78654000e+02],
             [0, 0, 0, 6.99943000e+02],
             [0, 0, 1/120., 0]])


def cam_homography(image, camera_k, RT=None):
    # inputs:
    #   image: ground image:
    #   camera_k: 3*3 K matrix of left color camera : 3*3
    #   RT: rotation and translate between cameras
    # return:
    #   out: image after RT

    # turn np to torch
    image = ToTensor(image)
    inverse_k = torch.from_numpy(np.linalg.inv(camera_k)).float()
    camera_k = torch.from_numpy(camera_k).float()

    #RT = np.hstack(((np.linalg.inv(RT[:,:3])), -RT[:,-1:]))
    RT = torch.from_numpy(RT).float()

    C, H, W = image.shape

    # get back warp matrix
    # meshgrid the target pannel
    i = torch.arange(0, H)
    j = torch.arange(0, W)
    ii, jj = torch.meshgrid(i, j)  # i:h,j:w
    uv1 = torch.stack([jj, ii, torch.ones_like(ii)], dim=-1).float()  # shape = [H, W, 3]

    # Trans matrix from target to real
    XYZ = torch.einsum('ij, hwj -> hwi', inverse_k, uv1) #[H, W, 3]
    ones = torch.ones_like(XYZ[:,:,0:1])*0.175# 0.1 0.125 0.15 - 0.2
    XYZ1 = torch.cat([XYZ, ones], dim=-1)  # [H,W,4]
    XYZ = torch.einsum('ij, hwj -> hwi', RT, XYZ1)  # [H,W,3]

    # project to camera
    uv1 = torch.einsum('ij, hwj -> hwi', camera_k, XYZ) # shape = [H,W,3]
    # only need view in front of camera ,Epsilon = 1e-6
    uv_last = torch.maximum(uv1[:, :, 2:], torch.ones_like(uv1[:, :, 2:]) * 1e-6)
    uv = uv1[:, :, :2] / uv_last  # shape = [H, W,2]

    # lefttop to center
    uv_center = uv - torch.tensor([W // 2, H // 2])  # shape = [H, W,2]
    scale = torch.tensor([W // 2, H // 2])
    uv_center /= scale

    out = F.grid_sample(image.unsqueeze(0), uv_center.unsqueeze(0), mode='bilinear',
                              padding_mode='zeros')

    if C == 1:
        mode = 'L'
    else:
        mode = 'RGB'
    out = transforms.functional.to_pil_image(out.squeeze(0),mode=mode)
    return out

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
    motion_file_on = os.path.join(root_dir, 'viso', "video_on_road_motion.mat")
    motion_dict_on = loadmat(motion_file_on)
    motion_file_off = os.path.join(root_dir, 'viso', "video_off_road_motion.mat")
    motion_dict_off = loadmat(motion_file_off)

    camera_k = mat
    bev_size = 720
    modes = ['train', 'test']
    for mode in modes:
        txt_path = os.path.join(root_dir, 'both_road_' + mode + '.txt')
        mask_file_names = load_dataset_path(txt_path)
        for mask_file_name in mask_file_names:
            # test on raod
            mask_file_name = '../Dataset/masks/on_road/left_mask_000000691.png'
            #mask_file_name = '../Dataset/masks/off_road/left_mask_000003972.png'

            file_num = int(re.findall('\d+', mask_file_name)[0])
            if 'on_road' in mask_file_name:
                fname = 'masks/on_road/left_mask_%09d.png' % (file_num)
                sub_dir = 'video_on_road'
                motion_dict = motion_dict_on

                #continue # for different rx
            else:
                fname = 'masks/off_road/left_mask_%09d.png' % (file_num)
                sub_dir = 'video_off_road'
                motion_dict = motion_dict_off

                #continue  # for different rx

            #process mask
            dir = os.path.join(save_dir, sub_dir)
            if not os.path.exists(dir):
                os.makedirs(dir)
            mask_name = os.path.join(dir, 'left_mask_%09d.png' % (file_num))

            fname = os.path.join(root_dir, fname)
            mask = cv2.imread(fname, 0)  # grey sclae

            # mask rectify # need??
            mask = cv2.remap(mask, map1x, map1y,
                                   cv2.INTER_CUBIC)
            mask = transforms.functional.to_pil_image(mask, mode='L')
            #
            # bev_mask = cam_to_bev(mask, camera_k, bev_size, RT=None)
            #
            # # save mask
            # bev_mask.save(mask_name)

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

                # turn previous camera image to current image
                left = cam_homography(frame_left, camera_k, RT=motion[:3])
                right = cam_homography(frame_right, camera_k, RT=motion[:3])

                if 1:
                    cv2.imwrite('left_ori_%d.png' % (file_id), frame_left)
                    cv2.imwrite('right_ori_%d.png' % (file_id), frame_right)
                    left.save('left_%d.png' % (file_id))
                    right.save('right_%d.png' % (file_id))

                # debug merge left and mask, right and mask
                if 1:
                    if file_id == file_num:
                        left_a = left.convert("RGBA")
                        right_a = right.convert("RGBA")
                        mask_a = mask.convert("RGBA")
                        merge_lm = Image.blend(left_a, mask_a, alpha=.5)
                        merge_lm.save('merge_lm.png')
                        merge_rm = Image.blend(right_a, mask_a, alpha=.5)
                        merge_rm.save('merge_rm.png')

                # debug merge left and right
                if 0:
                    left_a = left.convert("RGBA")
                    right_a = right.convert("RGBA")
                    merge_lr = Image.blend(left_a, right_a, alpha=.5)
                    merge_lr.save('merge_lr.png')

                # debug merge left sequence
                if 1:
                    if file_id == file_num:
                        merge_seq = left.convert("RGBA")
                    else:
                        left_a = left.convert("RGBA")
                        merge_seq = Image.blend(merge_seq, left_a, alpha=.5)
                        merge_seq.save('merge_seq_%d.png'%(file_id))

                # motion update
                if 'img_%09d'%(file_id) in motion_dict.keys():
                    motion_update = motion_dict['img_%09d'%(file_id)]

                    R = motion[:3,:3] @ np.linalg.inv(motion_update[:3,:3])
                    t = motion[:3,-1:] - (motion[:3,:3] @ motion_update[:3,-1:])

                    motion = np.vstack(( np.hstack((R,t)), motion[-1:]))

                # save rgb image
                rgb_dir = os.path.join(save_dir, sub_dir)
                save_image(left, rgb_dir, 'left', file_num, file_id)
                save_image(right, rgb_dir, 'right', file_num, file_id)

    print("bev trans finish")
