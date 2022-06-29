import sys
sys.path.append('/home/shan/projects/water_hazard/pyviso2/src')
import viso2
import cv2
import numpy as np
import os
import re
from scipy.io import savemat
import zedutils
# from mayavi import mlab
import open3d as o3d

root_dir = '/home/shan/Dataset/water_hazard/'#'/data/dataset/water_hazard/'
save_dir = 'viso'

map1x, map1y, map2x, map2y, mat, Q1 = \
                zedutils.getTransformFromConfig('SN1994.conf', Type='CAM_HD')
K_l, K_r, R_lr, T_lr= zedutils.getKRTInfo('SN1994.conf', Type='CAM_HD')
Max_point_cnt = 2000
plane_cnt = 3
default_d = -1.67

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

X_RADIUS = 2
Z_RADIUS = 5
Y_MIN = 1.4
Y_MAX = 2
def getCuboidPoints():
  return np.array([
    # Vertices Polygon1
    [X_RADIUS, Y_MAX, Z_RADIUS],
    [-X_RADIUS, Y_MAX, Z_RADIUS],
    [-X_RADIUS, Y_MIN, Z_RADIUS],
    [X_RADIUS, Y_MIN, Z_RADIUS],

    # Vertices Polygon 2
    [X_RADIUS, Y_MAX, -Z_RADIUS],
    [-X_RADIUS, Y_MAX, -Z_RADIUS],
    [-X_RADIUS, Y_MIN, -Z_RADIUS],
    [X_RADIUS, Y_MIN, -Z_RADIUS],
  ]).astype("float64")

if __name__ == '__main__':
    save_dir = os.path.join(root_dir, save_dir)
    dirs = ['video_on_road']#, 'video_off_road']
    for dir in dirs:
        viso, recon = viso_init()
        file_names = os.listdir(os.path.join(root_dir, dir))
        last_norm = [0,1,0,default_d]

        # order by number
        file_names = sorted(file_names)

        pose = viso2.Matrix_eye(4)
        motion_dict = {}
        norm_dict = {}
        for file_name in file_names:
            if 'ppm' not in file_name:
                continue

            file_num = int(re.findall('\d+', file_name)[0])
            # if file_num == 2464: # for debug
            #     viso, recon = viso_init()
            #     pose = viso2.Matrix_eye(4)

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
                motion = viso.getMotion() # p_t-1 -> p_t

                # save motion
                motion_np = np.zeros((4, 4))
                motion.toNumpy(motion_np)
                motion_dict[file_name[:13]] = motion_np

                est_motion = viso2.Matrix_inv(motion) #turn to frame 0 coordinate
                pose = pose * est_motion

                # # get tracked points and estimate norm of ground plane
                # track = recon.getTracks()
                #
                # # turn swig vector to pcd
                # if track.size() > 10:
                #     if track.size() > Max_point_cnt:
                #         len = Max_point_cnt
                #     else:
                #         len = track.size()
                #     pts = np.empty((len, 3))
                #     # use latest 500 points?
                #     for i, t in enumerate(track[-len:]):
                #         pts[i, :] = (t.pt.x, t.pt.y, t.pt.z)
                #
                #     # from 1st coordinate to current
                #     ori2cur = viso2.Matrix_inv(pose)
                #     ori2cur_np = np.zeros((4, 4))
                #     ori2cur.toNumpy(ori2cur_np)
                #     pts_cur = ori2cur_np @ np.vstack((pts.T,np.ones((1,len))))
                #     pts = pts_cur[:3].T
                #
                #     # Create an empty point cloud
                #     # Use pcd.point to access the points' attributes
                #     pcd = o3d.geometry.PointCloud()
                #     pcd.points = o3d.utility.Vector3dVector(pts)
                #
                #     while 1:
                #         plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                #                                                  ransac_n=5,
                #                                                  num_iterations=1000)
                #         [a, b, c, d] = plane_model
                #         print(f"point count: {track.size()} Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
                #         if abs(b) > 0.75 and abs(b-lastb) < 0.15 and (abs(d)<2 and abs(d)>1):
                #             # find the ground plane
                #             break
                #         # wrong norm, remove points and find another plane
                #         pcd = pcd.select_by_index(inliers, invert=True)
                #
                #     lastb = b
                #     norm_dict[file_name[:13]] = np.array([a,b,c,d])
                # else:
                #     norm_dict[file_name[:13]] = np.array([0,1,0,-1.77])

                # pose_np = np.zeros((4, 4))
                # pose.toNumpy(pose_np)
                # pose_dict[file_name[:13]] = pose_np

                num_matches = viso.getNumberOfMatches()
                num_inliers = viso.getNumberOfInliers()
                print('Matches:', num_matches, "Inliers:", 100 * num_inliers / num_matches, '%, Current pose:',pose)  # pose is 4*4 matrix
                matches = viso.getMatches()
                assert (matches.size() == num_matches)
                recon.update(matches, motion, 0)

                points = recon.getPoints()
                # turn swig vector to pcd
                if points.size() > plane_cnt*4:
                    if points.size() > Max_point_cnt:
                        len = Max_point_cnt
                    else:
                        len = points.size()
                    pts = np.empty((len, 3))
                    # use latest points
                    for i, p in enumerate(points[-len:]):
                        pts[i, :] = (p.x, p.y, p.z)

                    # from 1st coordinate to current
                    ori2cur = viso2.Matrix_inv(pose)
                    ori2cur_np = np.zeros((4, 4))
                    ori2cur.toNumpy(ori2cur_np)
                    pts_cur = ori2cur_np @ np.vstack((pts.T, np.ones((1, len))))
                    pts = pts_cur[:3].T

                    # Create an empty point cloud
                    # Use pcd.point to access the points' attributes
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)

                    # select points along z
                    if np.asarray(pcd.points).shape[0] > plane_cnt:
                        ## Start point here corresponds to an ego vehicle position start in a point cloud
                        print('point cnt: ', np.asarray(pcd.points).shape[0], ' before crop')
                        cuboid_points = getCuboidPoints() # return 8 corner points
                        box_points = o3d.utility.Vector3dVector(cuboid_points)
                        oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(box_points)
                        pcd = pcd.crop(oriented_bounding_box)
                        print('point cnt: ', np.asarray(pcd.points).shape[0], ' after crop')

                    while np.asarray(pcd.points).shape[0]>plane_cnt:
                        plane_model, inliers = pcd.segment_plane(distance_threshold=0.03,
                                                                 ransac_n=plane_cnt,
                                                                 num_iterations=1000)
                        [a, b, c, d] = plane_model
                        print(
                            f"point count: {points.size()} Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
                        if abs(b) > 0.8 and (d > -1.8 and d < -1.4):
                            # find the ground plane
                            norm_dict[file_name[:13]] = np.array([a, b, c, d])
                            last_norm = np.array([a, b, c, d])
                            break
                        # wrong norm, remove points and find another plane
                        pcd = pcd.select_by_index(inliers, invert=True)
                    else:
                        norm_dict[file_name[:13]] = last_norm
                else:
                    norm_dict[file_name[:13]] = last_norm

                if 0: #debug
                    if len(motion_dict.keys()) >= len(file_names)//10:
                        break
            else:
                print('.... failed!')
                # save pose
                pose_np = np.zeros((4, 4))
                pose.toNumpy(pose_np)
                norm_dict[file_name[:13]] = np.array([0,1,0,default_d])

        if 0: #debug
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
