import viso2
import numpy as np
import cv2
import os
from stereo_recon import getStereoDisparityPair
from ..camera_recorder import zedutils
from ..plane_fitting import fitPlane

def demoWarp3D(show=True):
#    Folder = '/home/chuong/workspace/pycamrecorder/video_21-08-2016_12-14-49'
    Folder = '/home/chuong/workspace/pycamrecorder/video_21-08-2016_14-21-39'
    odometry_file = os.path.join(Folder, 'video_21-08-2016_12-14-49_log.txt')
    baseLine = 120.0 #mm
    Tx = -baseLine
    step = 5 # 1 #
    IndexRange = range(500, 5000-step, step)

    map1x, map1y, map2x, map2y, mat, Q = \
        zedutils.getTransformFromConfig(FileName='SN1994.conf', Type='CAM_HD')
    print(mat)
    print(Q)
    Q = np.array([[1, 0, 0, -mat[0, 2]],
                  [0, 1, 0, -mat[1, 2]],
                  [0, 0, 0,  mat[0, 0]],
                  [0, 0, -1/Tx,      0]])
    print(Q)
    params = viso2.Stereo_parameters()
    params.calib.f  = mat[0, 0]
    params.calib.cu = mat[0, 2]
    params.calib.cv = mat[1, 2]
    params.base     = 1.0/Q[3,2] # mm
    print(params.calib.f, params.calib.cu, params.calib.cv, params.base)

    # initialize visual odometry
    viso = viso2.VisualOdometryStereo(params)
    recon = viso2.Reconstruction()
    recon.setCalibration(params.calib.f, params.calib.cu, params.calib.cv)

    pose = viso2.Matrix_eye(4)
    pose_np = np.zeros([4, 4])
    motion_np = np.zeros([4, 4])
    est_motion_np = np.zeros([4, 4])
    tvec_KF = np.array([0, 0, 0])
    alpha_KF = 0.7 # from 0.67 to 0.81
    for i in IndexRange: #[1760]: #[1080]: #[1340]: #IndexRange: #
        left_filename0  = os.path.join(Folder, 'Left_img_%09d_rec.png' % i)
        right_filename0 = os.path.join(Folder, 'Right_img_%09d_rec.png' % i)
        left_filename1  = os.path.join(Folder, 'Left_img_%09d_rec.png' % (i+step))
        right_filename1 = os.path.join(Folder, 'Right_img_%09d_rec.png' % (i+step))

        if os.path.exists(left_filename0):
            print('Found %s' % left_filename0)
            leftFrame0 = cv2.imread(left_filename0)
            rightFrame0 = cv2.imread(right_filename0)
            leftFrame1 = cv2.imread(left_filename1)
            rightFrame1 = cv2.imread(right_filename1)
            leftFrameGray1 = cv2.cvtColor(leftFrame1, cv2.COLOR_BGR2GRAY)
            rightFrameGray1 = cv2.cvtColor(rightFrame1, cv2.COLOR_BGR2GRAY)
            if viso.process_frame(leftFrameGray1, rightFrameGray1):
                motion = viso.getMotion()

                num_matches = viso.getNumberOfMatches()
                num_inliers = viso.getNumberOfInliers()
                print 'Matches:', num_matches, "Inliers:", 100*num_inliers/num_matches, '%, Current pose:'
                print(pose)
                matches = viso.getMatches()
                assert(matches.size() == num_matches)
                recon.update(matches, motion, 0)

                est_motion = viso2.Matrix_inv(motion)
                pose = pose * est_motion
                pose.toNumpy(pose_np)
                motion.toNumpy(motion_np)
                est_motion.toNumpy(est_motion_np)
                rvec, _ = cv2.Rodrigues(est_motion_np[:3, :3])
                tvec = est_motion_np[:3, 3]
                tvec_KF = alpha_KF*tvec_KF + (1 - alpha_KF)*tvec
#                rvec, _ = cv2.Rodrigues(motion_np[:3, :3])
#                tvec = motion_np[:3, 3]
                print(tvec)

                # computer stereo disparity
                disp0 = getStereoDisparityPair(leftFrame0, rightFrame0,
                                                            window_size=19,
                                                            dispRange=[0, 32])
                print(disp0.min(), disp0.max())
                u, v = np.meshgrid(np.arange(disp0.shape[1]),
                                   np.arange(disp0.shape[0]))
                plane_coefs0, _ = fitPlane(u, v, disp0, infHeight=400, dispMax=25.0,
                                          skip=5, useTriangle=True)
    #            showPlane(u, v, disp0, plane_coefs0)
    #            plt.show()

                # perform image warping by projection and backprojection
#                warp3D(leftFrame0, leftFrame1, disp0, plane_coefs0, Q, mat,
#                       rvec, tvec, show=True)
                leftFrame0_warped = warp3D(leftFrame0, disp0,
                                           plane_coefs0, Q, mat,
                                           rvec, tvec_KF)
                image_diff = np.abs(leftFrame0_warped.astype(np.float32) -
                                    leftFrame1.astype(np.float32))
                image_diff[leftFrame0_warped == 0] = 0
                if show:
                    cv2.imshow('image0', leftFrame0[::2,::2])
                    cv2.imshow('image0_warped', leftFrame0_warped[::2,::2])
                    cv2.imshow('image1', leftFrame1[::2,::2])
                    cv2.imshow('image0_warped-image1', 5*image_diff[::2,::2].astype(np.uint8))
                    cv2.imshow('disparity', disp0[::2,::2]/disp0.max())
                    key = cv2.waitKey(10) & 255
                    if key == ord(' '):
                        print('Press any key to continue...')
                        cv2.waitKey(0)
                    if key == ord('q'):
                        break

if __name__ == '__main__':
    demoWarp3D()
