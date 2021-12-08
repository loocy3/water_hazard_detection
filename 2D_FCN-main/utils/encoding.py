import numpy as np
import utils.zedutils

# In[2]:
def rebin(arr, new_shape):
    '''
    Source: https://scipython.com/blog/binning-a-2d-array-in-numpy/
    '''
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def rebin_image2D(arr, bin_size):
    '''
    Modified version of rebin to accept bin_size instead of new_shape
    '''
    shape = (arr.shape[0] // bin_size[0], bin_size[0],
             arr.shape[1] // bin_size[1], bin_size[1])
    return arr.reshape(shape).mean(axis=(-1, 1))


def rebin_image3D(img, bin_size=(2, 2)):
    '''
    Bin images according to bin_size
    '''
    img_bin = np.stack([rebin_image2D(img[:, :, i].astype(np.float32), bin_size) for i in range(img.shape[2])],
                       axis=2)

    return img_bin

#One hot encoding for each pixel
def one_hot(Y):
    result = []
    for i in range(len(Y)):
        img_array = []
        for j in range(len(Y[i])):
            e = Y[i][j]
            if e == 255:
                img_array.append(np.array([0,1]))
            elif e == 0:
                img_array.append(np.array([1,0]))
        result.append(img_array)
    return np.array(result)