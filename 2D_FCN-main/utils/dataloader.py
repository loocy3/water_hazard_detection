import re
import cv2
import os
from utils.transformation import logTransformImage
import numpy as np
from utils.encoding import rebin_image2D, rebin_image3D, one_hot
from utils import zedutils


def load_dataset_path(txt_path):
    p = np.genfromtxt(txt_path, dtype='str')
    return p[:, 1]

def generate_dataset(WORK_ON):
    outfile = os.path.join('Dataset/' + WORK_ON + '_data.npz')
    if os.path.exists(outfile) == False:
        print('Read ' + 'Dataset/' + WORK_ON + '_train.txt')
        X_train, y_train = load_data_bothroad('Dataset/video_' + WORK_ON, 'Dataset/' + WORK_ON + '_train.txt',
                                              resize_by=(3, 4))
        print('Read ' + 'Dataset/' + WORK_ON + '_test.txt')
        X_test, y_test = load_data_bothroad('Dataset/video_' + WORK_ON, 'Dataset/' + WORK_ON + '_test.txt',
                                            resize_by=(3, 4))
        print('Convert to one hot')
        y_train = one_hot(y_train)
        y_test = one_hot(y_test)
        print('Saved precomputed Dataset to %s' % outfile)
        np.savez(outfile, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    else:
        print('Load precomputed Dataset file %s' % outfile)
        npzfile = np.load(outfile)
        X_train = npzfile['X_train']
        y_train = npzfile['y_train']
        X_test = npzfile['X_test']
        y_test = npzfile['y_test']
    return X_train, y_train, X_test, y_test

def load_data_bothroad(img_folder, txt_path, resize_by, num_frames=8, skipping=False):
    # load ZED camera config and
    # obtain stereo rectification maps, camera matrix and Q matrix
    # default is skipping to
    map1x, map1y, map2x, map2y, mat, Q = zedutils.getTransformFromConfig('SN1994.conf', Type='CAM_HD')
    stereo = cv2.StereoSGBM_create(0, 32, 19, speckleWindowSize=100, speckleRange=2)

    X = []
    Y = []

    mask_file_names = load_dataset_path(txt_path)

    for mask_file_name in mask_file_names:
        file_num = int(re.findall('\d+', mask_file_name )[0])
        if 'on_road' in mask_file_name:
            fname = 'Dataset/masks/on_road/left_mask_%09d.png' % (file_num)
            img_folder = 'Dataset/video_on_road/'
        else:
            fname = 'Dataset/masks/off_road/left_mask_%09d.png' % (file_num)
            img_folder = 'Dataset/video_off_road/'
        mask = cv2.imread(fname, 0)
        if skipping:
            mask = mask[::resize_by[0], ::resize_by[1]]
        else:
            mask = np.uint8(rebin_image2D(mask, resize_by))
            mask[mask>127] = 255
            mask[mask<=127] = 0
        #load mask to Y
        Y.append(mask.flatten())
        #load num_frames before truth mask.
        video = []
        i = 0
        for j in range(file_num-num_frames+1,file_num+1):
            image_file_name = os.path.join(img_folder, 'img_%09d.ppm' % j)
            #load pair
            pair = cv2.imread(image_file_name)
            pair = logTransformImage(pair)
            #load only left frame.
            frame_left = pair[:,:pair.shape[1]//2,:].copy()
            #remap
            frame_left_rec = frame_left #cv2.remap(frame_left, map1x, map1y, cv2.INTER_CUBIC)
            #resize
            if skipping:
                frame_left_rec = frame_left_rec[::resize_by[0], ::resize_by[1], :]
            else: # binning
                frame_left_rec = np.uint8(rebin_image3D(frame_left_rec, resize_by))

            #images form a video
            video.append(frame_left_rec)

        X.append(video)

    return np.array(X),np.array(Y)# In[5]: