import viso2
import cv2
import numpy as np
import os
import re
from scipy.io import savemat
import zedutils
from mayavi import mlab
from BEV_transform_withPose import cam_to_bev
from PIL import Image
import open3d as o3d

root_dir = '/data/dataset/water_hazard/'
save_dir = 'viso'

map1x, map1y, map2x, map2y, mat, Q1 = \
                zedutils.getTransformFromConfig('SN1994.conf', Type='CAM_HD')
K_l, K_r, R_lr, T_lr= zedutils.getKRTInfo('SN1994.conf', Type='CAM_HD')

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

if __name__ == '__main__':
    save_dir = os.path.join(root_dir, save_dir)
    dirs = ['video_off_road']#,'video_off_road']
    for dir in dirs:
        viso, recon = viso_init()
        file_names = os.listdir(os.path.join(root_dir, dir))

        # order by number
        file_names = sorted(file_names)

        last_bev = None
        pose = viso2.Matrix_eye(4)
        motion_dict = {}
        norm_dict = {}
        for file_name in file_names:
            if 'ppm' not in file_name:
                continue

            file_num = int(re.findall('\d+', file_name)[0])
            # if file_num == 2464: # for debug
            #     viso, recon = viso_init()

            full_file_name = os.path.join(root_dir, dir, file_name)
            pair = cv2.imread(full_file_name)
            frame_left = pair[:, :pair.shape[1] // 2, :].copy()
            frame_right = pair[:, pair.shape[1] // 2:, :].copy()

            # image rectify
            frame_left = cv2.remap(frame_left, map1x, map1y,
                                cv2.INTER_CUBIC)
            frame_right = cv2.remap(frame_right, map2x, map2y,
                                cv2.INTER_CUBIC)
            # turn grb to grey
            left_grey = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            right_grey = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            if viso.process_frame(left_grey, right_grey):
                motion = viso.getMotion() # p_t-1 -> p_t or p_t -> p_t-1

                # save motion
                motion_np = np.zeros((4, 4))
                motion.toNumpy(motion_np)
                motion_dict[file_name[:13]] = motion_np

                est_motion = viso2.Matrix_inv(motion) #p_t -> p_t-1 so inv
                #est_motion = motion # p_t-1 -> p_t
                pose = pose * est_motion

                # get inliner points and estimate norm of ground plane
                inliers = viso.getInlierIndices()

                # turn swig vector to pcd
                pts = np.empty((inliers.size(), 3))
                for i, p in enumerate(points):
                    pts[i, :] = (p.x, p.y, p.z)

                pcd = o3d.t.geometry.PointCloud()
                pcd.point["positions"] = o3d.core.Tensor(pts)
                plane_model, _ = pcd.segment_plane(distance_threshold=0.03,
                                                         ransac_n=5,
                                                         num_iterations=1000)
                [a, b, c, d] = plane_model
                print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
                norm_dict[file_name[:13]] = np.array([a,b,c])

                # pose_np = np.zeros((4, 4))
                # pose.toNumpy(pose_np)
                # pose_dict[file_name[:13]] = pose_np

                num_matches = viso.getNumberOfMatches()
                num_inliers = viso.getNumberOfInliers()
                print('Matches:', num_matches, "Inliers:", 100 * num_inliers / num_matches, '%, Current pose:',pose)  # pose is 4*4 matrix
                matches = viso.getMatches()
                assert (matches.size() == num_matches)
                recon.update(matches, motion, 0)

                if 1: #debug
                    if len(pose_dict.keys()) >= len(file_names)//10:
                        break
            else:
                print('.... failed!')
                # save pose
                pose_np = np.zeros((4, 4))
                pose.toNumpy(pose_np)
                pose_dict[file_name[:13]] = pose_np

        if 1: #debug
            points = recon.getPoints()
            print("Reconstructed", points.size(), "points...")

            pts = np.empty((points.size(), 3))
            for i, p in enumerate(points):
                pts[i, :] = (p.x, p.y, p.z)

            mlab.figure()
            mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], colormap='copper')
            mlab.show()

        # save motion file
        motion_file = os.path.join(save_dir, dir+"_motion.mat")
        savemat(motion_file, motion_dict)
        # pose_file = os.path.join(save_dir, dir+"_pose.mat")
        # savemat(pose_file, pose_dict)
        norm_file = os.path.join(save_dir, dir+"_norm.mat")
        savemat(norm_file, norm_dict)

    print("save motion finish")
