
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


ToTensor = transforms.Compose([
    transforms.ToTensor()])

root_dir = '/data/dataset/water_hazard/'
save_dir = 'viso_homo'
sequence_len = 16
camera_height = 1.77


map1x, map1y, map2x, map2y, mat, Q1 = \
                zedutils.getTransformFromConfig('SN1994.conf', Type='CAM_HD')
K_l, K_r, R_lr, T_lr= zedutils.getKRTInfo('SN1994.conf', Type='CAM_HD')

Q = np.array([[1, 0, 0, -6.57925000e+02],
             [0, 1, 0, -3.78654000e+02],
             [0, 0, 0, 6.99943000e+02],
             [0, 0, 1/120., 0]])

def load_dataset_path(txt_path):
    p = np.genfromtxt(txt_path, dtype='str')
    return p[:, 1]

def cam_homography(image, camera_k, RT=None, tar_R=None, pitch=87.5):
    # inputs:
    #   image: ground image:
    #   camera_k: 3*3 K matrix of left color camera : 3*3
    #   RT: rotation and translate between cameras
    #   tar_R: rotation of target camera
    # return:
    #   out: image after RT

    # calculate n_vec
    n_vec = np.array([0,-1,0]) # y axis
    pitch = pitch*np.pi/180 # off 87 on:87.5 #on 3881
    n_vec = np.array([0, -np.sin(pitch), -np.cos(pitch)])
    n_vec = tar_R@n_vec[:,None] #[3,1]

    # H = K(R-tn/d)inv(K)
    R = RT[:3,:3]
    T = RT[:3,-1:]
    H = camera_k@(R-T@(n_vec.T)/camera_height)@np.linalg.inv(camera_k)

    out = cv2.warpPerspective(image, H, (image.shape[1],image.shape[0]))
    # to Image
    out = Image.fromarray(out.astype('uint8'), 'RGB')
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

    # read motion(from cur->previous) & pose
    motion_file_on = os.path.join(root_dir, 'viso', "video_on_road_motion.mat")
    motion_dict_on = loadmat(motion_file_on)
    motion_file_off = os.path.join(root_dir, 'viso', "video_off_road_motion.mat")
    motion_dict_off = loadmat(motion_file_off)
    pose_file_on = os.path.join(root_dir, 'viso', "video_on_road_pose.mat")
    pose_dict_on = loadmat(pose_file_on)
    pose_file_off = os.path.join(root_dir, 'viso', "video_off_road_pose.mat")
    pose_dict_off = loadmat(pose_file_off)

    camera_k = mat
    # bev_size = 720
    modes = ['train', 'test']
    for mode in modes:
        txt_path = os.path.join(root_dir, 'both_road_' + mode + '.txt')
        mask_file_names = load_dataset_path(txt_path)
        for mask_file_name in mask_file_names:
            # test on raod
            # mask_file_name = '../Dataset/masks/on_road/left_mask_000000691.png'
            #mask_file_name = '../Dataset/masks/off_road/left_mask_000003972.png'
            #mask_file_name = '../Dataset/masks/on_road/left_mask_000003881.png'

            file_num = int(re.findall('\d+', mask_file_name)[0])
            if 'on_road' in mask_file_name:
                fname = 'masks/on_road/left_mask_%09d.png' % (file_num)
                sub_dir = 'video_on_road'
                motion_dict = motion_dict_on
                pose_dict = pose_dict_on
                pitch = 87.5
            else:
                fname = 'masks/off_road/left_mask_%09d.png' % (file_num)
                sub_dir = 'video_off_road'
                motion_dict = motion_dict_off
                pose_dict = pose_dict_off
                pitch = 87

            # # exist continue
            # left_saved_dir = os.path.join(save_dir, sub_dir, 'left', 'img_%09d' % (file_num))
            # if os.path.exists(left_saved_dir):
            #     continue

            # # anchor -1, the last image. pose -1
            # all_file = sorted(pose_dict.keys())
            # anchor_pose = pose_dict[all_file[-1]]
            # anchor_R = anchor_pose[:3,:3]


            # process sequence of rgb
            motion = np.eye(4)
            # no cur_R, because ground plane also change rotation
            #cur_R = pose_dict['img_%09d'%(file_num)][:3,:3] @ anchor_R.T
            cur_R = np.eye(3) #
            for file_id in range(file_num,file_num-sequence_len,-1):
                rgb_file_name = os.path.join(root_dir, sub_dir, 'img_%09d.ppm'%(file_id))
                pair = cv2.imread(rgb_file_name)
                if pair is None:
                    continue
                pair = pair[..., ::-1]  # grb to rgb
                frame_left = pair[:, :pair.shape[1] // 2, :].copy()
                frame_right = pair[:, pair.shape[1] // 2:, :].copy()

                # image rectify # need??
                frame_left = cv2.remap(frame_left, map1x, map1y,
                                       cv2.INTER_CUBIC)
                frame_right = cv2.remap(frame_right, map2x, map2y,
                                        cv2.INTER_CUBIC)

                # turn previous camera image to current image
                left = cam_homography(frame_left, camera_k, RT=motion[:3], tar_R=cur_R, pitch=pitch)
                right = cam_homography(frame_right, camera_k, RT=motion[:3], tar_R=cur_R, pitch=pitch)

                if 0:
                    cv2.imwrite('left_ori_%d.png' % (file_id), frame_left)
                    cv2.imwrite('right_ori_%d.png' % (file_id), frame_right)
                    left.save('left_%d.png' % (file_id))
                    right.save('right_%d.png' % (file_id))

                # debug merge left sequence
                if 0:
                    if file_id == file_num:
                        merge_seq = left.convert("RGBA")
                    else:
                        left_a = left.convert("RGBA")
                        merge_seq = Image.blend(merge_seq, left_a, alpha=.5)
                        merge_seq.save('merge_seq_%d.png'%(file_id))

                # motion update
                if 'img_%09d'%(file_id) in motion_dict.keys():
                    motion_update = motion_dict['img_%09d'%(file_id)]

                    R = motion[:3,:3] @ motion_update[:3,:3]#np.linalg.inv(motion_update[:3,:3])
                    t = motion[:3,-1:] + (motion[:3,:3].T @ motion_update[:3,-1:])
                    motion = np.vstack((np.hstack((R,t)), motion[-1:]))

                # save rgb image
                rgb_dir = os.path.join(save_dir, sub_dir)
                save_image(left, rgb_dir, 'left', file_num, file_id)
                save_image(right, rgb_dir, 'right', file_num, file_id)

    print("bev trans finish")
