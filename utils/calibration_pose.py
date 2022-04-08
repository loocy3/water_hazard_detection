
import cv2
import numpy as np
import os
from scipy.io import loadmat,savemat

root_dir = '../../../Dataset/'
save_dir = 'viso'

if __name__ == '__main__':
    save_dir = os.path.join(root_dir, save_dir)

    # read motion
    motion_file_on = os.path.join(save_dir, "video_on_road_motion.mat")
    motion_dict_on = loadmat(motion_file_on)
    motion_file_off = os.path.join(save_dir, "video_off_road_motion.mat")
    motion_dict_off = loadmat(motion_file_off)

    dirs = ['video_on_road', 'video_off_road']
    for dir in dirs:
        file_names = os.listdir(os.path.join(root_dir, dir))

        # order by number
        file_names = sorted(file_names)

        # find index of calibrated image
        if dir == 'video_on_road':
            motion_dict = motion_dict_on
            cali_idx = file_names.index('img_000003060.ppm')
            R_cali, _ = cv2.Rodrigues(np.array([0.022, 0, 0.00298545]))
            # mask_file_name = '../Dataset/masks/on_road/left_mask_000000691.png'
        else:
            motion_dict = motion_dict_off
            cali_idx = file_names.index('img_000005551.ppm')
            R_cali, _ = cv2.Rodrigues(np.array([0.034, 0, 0.00298545]))
            # mask_file_name = '../Dataset/masks/off_road/left_mask_000003972.png'

        cali_r_dict = {}
        for file_name in file_names[cali_idx+1:]:
            if 'ppm' not in file_name:
                continue
            if file_name[:13] in motion_dict.keys():
                motion_update = motion_dict[file_name[:13]]
                cali_r_dict[file_name[:13]] = R_cali @ motion_update[:3,:3]

        motion_update = np.eye(3)
        before_file_names = file_names[:cali_idx+1]
        before_file_names.reverse()
        for file_name in before_file_names:
            if 'ppm' not in file_name:
                continue
            cali_r_dict[file_name[:13]] = R_cali @ np.linalg.inv(motion_update[:3, :3])
            if file_name[:13] in motion_dict.keys():
                motion_update = motion_dict[file_name[:13]]

        # save r file
        cali_r_file = os.path.join(save_dir, dir+"_r.mat")
        savemat(cali_r_file, cali_r_dict)

    print("save motion finish")
