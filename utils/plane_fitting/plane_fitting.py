# Chuong Nguyen 2016
# Water detection project
#
# Helper functions for plane fitting, horizon estimate, reflection angle, etc
# Author: Chuong Nguyen <chuong.nguyen@anu.edu.au>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
import os
from camera_recorder import zedutils


def planeFunctionCorr(x, im0, im1, mask, u, v):
    '''Compute cross correlation of left and warped right images given the disparity coefficients x.
    Calculate cross correlation of only pixels inside the mask.
    '''
    d = x[0]*u + x[1]*v + x[2]*1000
    d[d < 0] = 0
    u2 = u - d
    im1_warped = cv2.remap(im1, u2.astype(np.float32), v.astype(np.float32),
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    ret = cv2.matchTemplate(im0, im1_warped, method=cv2.TM_CCORR_NORMED, mask=mask)
    # print(ret[0,0])
    return 1 - ret[0,0]


def planeFunction(x, t, y, v0=None):
    '''Calculate error of data and plane equation.
    Calculate error of only pixels below line of infinity.
    '''
    if 0:
        # select points below vertical position v0
        mask = t[1] > v0
    else:
        # select points below vertical position v0
        mask0 = t[1] > v0
        # select points with positive y
        mask1 = y > 0
        mask = mask0*mask1
    diff = np.ravel(x[0]*t[0][mask] + x[1]*t[1][mask] + x[2] - y[mask])
#    diff[np.abs(diff) > 1.5] = 0
    return diff


def lineBtwPoints(p0, p1, x):
    ''' Function of a line connecting points pt0 and pt1.
    Given x value, return y value'''
    y = float(p1[1] - p0[1])/(p1[0] - p0[0])*(x - p0[0]) + p0[1]
    return y


def getHorizonLine(coefs, imageWidth, dHor=0.0):
    '''' This is where this parity is zeros or:
    dHor = u*coefs[0] + v*coefs[1] + coefs[2]'''
    u = np.arange(imageWidth)
    v = (dHor - u*coefs[0] - coefs[2])/coefs[1]
    return u, v

def fitPlaneCorr(groundCoefs, leftBGR, rightBGR, mask, useTriangle=True):
    '''

    Parameters
    ----------
    groundCoefs : TYPE
        DESCRIPTION.
    leftBGR : TYPE
        DESCRIPTION.
    rightBGR : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    leftGray = cv2.cvtColor(leftBGR, cv2.COLOR_BGR2GRAY)
    rightGray = cv2.cvtColor(rightBGR, cv2.COLOR_BGR2GRAY)
    scale = 1 #0.25 # 0.5 #
    leftGray = cv2.resize(leftGray, (0,0), fx=scale, fy=scale)
    rightGray = cv2.resize(rightGray, (0,0), fx=scale, fy=scale)
    mask = cv2.resize(np.uint8(mask), (0,0), fx=scale, fy=scale).astype(np.bool)
    u, v = np.meshgrid(np.arange(leftGray.shape[1]), np.arange(leftGray.shape[0]))
    if useTriangle:
        xMin = u.min()
        xMax = u.max()
        Height = v.max()
        # triangle region is below lines connecting these 3 points
        # or having larger v-values
        shrinkRatio = 0 # 0.15
        infHeight=leftGray.shape[0]//2
        pt0 = [xMin+shrinkRatio*xMax, Height]
        pt1 = [(xMin+xMax)/2, infHeight]
        pt2 = [xMax-shrinkRatio*xMax, Height]
        mask0 = v > lineBtwPoints(pt0, pt1, u)
        mask1 = v > lineBtwPoints(pt1, pt2, u)
        mask = mask0 * mask1
        # mask = mask * mask0 * mask1

    #cv2.imshow('mask', 255*np.uint8(mask))

    res_robust = least_squares(planeFunctionCorr, groundCoefs, loss='cauchy', f_scale=0.1,
                               args=(leftGray, rightGray, np.uint8(mask), u, v), verbose=0)
    a, b, c = res_robust.x

    return [a, b, c]


def fitPlane(u, v, disp, infHeight=400, dispMax=25.0, skip=5,
             useTriangle=True):
    ''' This is to fit a plane to disparity map of a plane.
        Plane is expressed as z = a*u +b*v + c
    Inputs:
        u, v: horizontal and vertical pixel coordinate
        disp: disparity map
        infHeight: approximated vertical pixel position of line of infinity
        dispMax: approximated max disparity at the bottom of image
        skip: skipping step in each direction to speedup calculation
        useTriangle: fit to only triangle region in front of the car
    Output:
        np.array([a, b, c])
    '''

    # values of normalise image width, height ahd disparity
    w, h, dmax = 1.0, 1.0, 1.0
    # rename disparity
    d = disp

    uu = u[::skip, ::skip]
    vv = v[::skip, ::skip]
    dd = d[::skip, ::skip]

    if useTriangle:
        xMin = u.min()
        xMax = u.max()
        Height = v.max()
        # triangle region is below lines connecting these 3 points
        # or having larger v-values
        shrinkRatio = 0 # 0.15
        pt0 = [xMin+shrinkRatio*xMax, Height]
        pt1 = [(xMin+xMax)/2, infHeight]
        pt2 = [xMax-shrinkRatio*xMax, Height]
        mask0 = vv > lineBtwPoints(pt0, pt1, uu)
        mask1 = vv > lineBtwPoints(pt1, pt2, uu)
        mask = mask0 * mask1
    else:
        mask = (vv > infHeight)*1
    uu = uu[mask]
    vv = vv[mask]
    dd = dd[mask]

    # normalisation
    uu = w*uu/d.shape[1]
    vv = h*vv/d.shape[0]
    dd = dmax*dd/dispMax

    # calculate initial conditions
    v0 = h*infHeight/d.shape[0] #
    u0, v0, d0 = [w/2.0, v0, 0.0]
    u1, v1, d1 = [w/2.0, h, dmax]
    u2, v2, d2 = [0.0, v0, 0.0]
    a = -d1*(v2-v0)/u0/(v1-v0)
    b = d1/(v1-v0)
    c = d1*(v2-2.0*v0)/(v1-v0)

    x0 = np.array([a, b, c])
    t_train = [uu, vv]
    y_train = dd
    res_robust = least_squares(planeFunction, x0, loss='cauchy', f_scale=0.1,
                               args=(t_train, y_train, v0), verbose=0)
    coefs = res_robust.x

    # convert to normal image width, height and disparity
    coefs = res_robust.x*np.array([dispMax/d.shape[1], dispMax/d.shape[0], dispMax])
    return coefs, mask

def getInliers(u, v, z, disp, tolerance=np.array([1, 0.1])):
    ''' Select inliers of plane fitting.
    Tolerance[0] is proportional to the noise of disparity map
    Tolerance[1] is proportional to the unevenness of the ground.
    There's is a trade off between increasing inliers and detecting
    distance objects.'''
    diff = np.abs(disp - z)
    mask = diff - tolerance[0] < tolerance[1]*disp
    return mask

def showPlane(u, v, d, coefs, showInliers=True, showOutliers=True, skip=10, live=True):
    # get a coarse data set
    uu = u[::skip,::skip]
    vv = v[::skip,::skip]
    dd = d[::skip,::skip]
    if showInliers or showOutliers:
        zz = coefs[0]*uu + coefs[1]*vv + coefs[2]
#        mask = getInliers(uu, vv, zz, dd)
        mask = dd > 2
        mask2 = np.logical_not(mask)

    # compute horizon line
    uHor, vHor = getHorizonLine(coefs, d.shape[1])

    if live:
        plt.ion()
    fig = plt.figure(100)
#    fig = plt.figure(figsize=(12.80, 7.20))
    ax = fig.add_subplot(111, projection='3d')

    # disparity inliers and outliers
    if showInliers:
        ax.scatter(uu[mask], vv[mask], dd[mask], color='red', marker='.')
    if showOutliers:
        ax.scatter(uu[mask2], vv[mask2], dd[mask2], color='blue', marker='.')

    # four corners of fitted plane
    uu2, vv2 = np.meshgrid([0, u.max()], [0, v.max()])
    vv2[0, 0] = vHor[0]
    vv2[0, 1] = vHor[-1]
    zz2 = coefs[0]*uu2 + coefs[1]*vv2 + coefs[2]
    ax.plot_surface(uu2, vv2, zz2, linewidths=3, alpha=0.6, color=[0,1,0])

    # fitting triangle
    xMin = u.min()
    xMax = u.max()
    Height = v.max()
    infHeight = vHor[d.shape[1]//2]

    # draw horizon line
    dHor = np.zeros_like(uHor)
    plt.plot(uHor, vHor, dHor, 'r', linewidth=3)

    # triangle region is below lines connecting these 3 points
    # or having larger v-values
    shrinkRatio = 0 # 0.15
    pt0 = [xMin+shrinkRatio*xMax, Height]
    pt1 = [(xMin+xMax)//2, infHeight]
    pt2 = [xMax-shrinkRatio*xMax, Height]
    x = [pt0[0], pt1[0], pt2[0]]
    y = [pt0[1], pt1[1], pt2[1]]
    z = [coefs[0]*pt0[0] + coefs[1]*pt0[1] + coefs[2],
         coefs[0]*pt1[0] + coefs[1]*pt1[1] + coefs[2],
         coefs[0]*pt2[0] + coefs[1]*pt2[1] + coefs[2]]
#    verts = [zip(x, y, z)]
#    collection = Poly3DCollection(verts, linewidths=3, alpha=0.6, color=[0,0,0])
#    ax.add_collection3d(collection)
    ax.plot([pt0[0], pt1[0], pt2[0], pt0[0]], [pt0[1], pt1[1], pt2[1], pt0[1]],
             [coefs[0]*pt0[0] + coefs[1]*pt0[1] + coefs[2],
              coefs[0]*pt1[0] + coefs[1]*pt1[1] + coefs[2],
              coefs[0]*pt2[0] + coefs[1]*pt2[1] + coefs[2],
              coefs[0]*pt0[0] + coefs[1]*pt0[1] + coefs[2]], 'k', linewidth=3)

    ax.invert_yaxis()
    ax.set_xlabel('u [pixel]', fontsize=16)
    ax.set_ylabel('v [pixel]', fontsize=16)
    ax.set_zlabel('Disparity [pixel]', fontsize=16)
    plt.tight_layout()
#    plt.savefig('plane_fitting.png')
    plt.show()

def getRotation(n1_, n2_, angleAxis=False):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    n1 = n1_/np.linalg.norm(n1_)
    n2 = n2_/np.linalg.norm(n2_)

    v = np.cross(n1, n2)
    s = np.linalg.norm(v)
    c = np.dot(n1, n2)
    I = np.eye(3)
    vx = np.asarray([[0,    -v[2], v[1]],
                     [v[2],  0,   -v[0]],
                     [-v[1], v[0], 0   ]])
    R = I + vx + (vx @ vx)/(1+c)
    if angleAxis:
        return cv2.Rodrigues(R)[0]
    return R

def getTransformMap(plane_coefs, disp, Q, ROIXZ=[-10., 10., 0.5, 20.],
                    outputSize=[320, 240]):

    # get camera intrinscis and Tx
    # https://answers.opencv.org/question/187734/derivation-for-perspective-transformation-matrix-q/
    u0, v0, f, Tx = -Q[0, 3], -Q[1, 3], Q[2, 3], -1.0/Q[3, 2]
    # plane coefficients of disparity d = a*u + b*v + c
    a, b, c = plane_coefs

    # reprojectImageTo3D will generate point cloud [x,y,z] on a plane
    # a*x + b*y + (a*u0 + b*v0 + c)/f*z + Tx = 0
    # normal vector of the plane is
    n1 = np.asarray([a, b, (a*u0 + b*v0 + c)/f])
    y0 = abs(Tx/np.linalg.norm(n1)) # distance from camera origin to plane
    n1 = n1/np.linalg.norm(n1)

    # n1, y0 = getPlaneNormalDistance(disp, Q)

    # we want to rotate this plane to plane y = y0
    # normal vector of this plane is
    n2 = np.asarray([0, 1, 0])
    #print(n1, n2, y0)

    R12 = getRotation(n1, n2)
    R21 = getRotation(n2, n1)

    # get bird view by generating grids on x-z plane
    x_range = np.linspace(ROIXZ[0], ROIXZ[1], outputSize[0])
    z_range = np.linspace(ROIXZ[2], ROIXZ[3], outputSize[1])[::-1]
    x, z = np.meshgrid(x_range, z_range)

    # convert to mm
    x = x*1000
    z = z*1000

    # y coordinates equal to camera distance to plane
    y = y0*np.ones_like(x)

    xz = np.vstack([x.reshape([1,-1]), y.reshape([1,-1]), z.reshape([1,-1])])
    xz_rot = R21 @ xz

    # compute u, v coordinates in normal camera image
    u = u0 + f*xz_rot[0]/xz_rot[2]
    v = v0 + f*xz_rot[1]/xz_rot[2]
    u = u.reshape(x.shape)
    v = v.reshape(x.shape)
    #print('u value: ', u)
    #print('v value: ', v)
    return u, v


def getPlaneNormalDistance(disp, Q):
    xyz = cv2.reprojectImageTo3D(disp, Q)
    xs = xyz[:,:,0].flatten()
    ys = xyz[:,:,1].flatten()
    zs = xyz[:,:,2].flatten()
    d = disp.flatten()
    xs = xs[d > 0]
    ys = ys[d > 0]
    zs = zs[d > 0]
    ones = np.ones_like(xs)

    # do fit equation a*x + b*y + c = z
    b = zs.reshape(-1,1)
    A = np.hstack([xs.reshape(-1,1), ys.reshape(-1,1), ones.reshape(-1,1)])

    # Manual solution
    fit = np.linalg.inv(A.T @ A) @ A.T @ b
    errors = b - A @ fit
    # residual = np.linalg.norm(errors)

    # plane equation a*x + b*y + c*z + d = 0
    a, b, c, d = fit[0,0], fit[1,0], -1, fit[2,0]
    n = np.asarray([a, b, c])
    y0 = d/np.linalg.norm(n)
    n = n/np.linalg.norm(n)
    if n.sum() < 0:
        n = -n

    return n, y0

def track3D(image, mask, plane_coefs, disp, Q, ROIXZ=[-10., 10., 0.5, 20.], showAxes=False, show=False):
    u, v = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    d = plane_coefs[0]*u + plane_coefs[1]*v + plane_coefs[2]

    # # horizon line
    # uHor, vHor = getHorizonLine(plane_coefs, image.shape[1])
    d[d < 0] = 0
    xyz = cv2.reprojectImageTo3D(d.astype(np.float32), Q)

    mask_x = xyz[:, :, 0][mask == 0]
    mask_y = xyz[:, :, 1][mask == 0]
    mask_z = xyz[:, :, 2][mask == 0]
    mask_d = d[mask == 0]

    mask_xyz = np.vstack((mask_x, mask_y, mask_z)).T

    # get bird view by a simple dewarp ground image and ignoring y component
    # width = 720
    # height = 1280
    width = 320
    height = 240
    # x_range = np.linspace(ROIXZ[0], ROIXZ[1], width)
    # z_range = np.linspace(ROIXZ[2], ROIXZ[3], height)[::-1]
    # x, z = np.meshgrid(x_range, z_range)
    # # convert to mm
    # x = x*1000
    # z = z*1000
    # u0, v0, f, Tx = -Q[0, 3], -Q[1, 3], Q[2, 3], -1.0/Q[3, 2]
    # u = x/z*f + u0
    # d = -(u - u0 - f)/(x - z)*Tx
    # v = (d - plane_coefs[0]*u - plane_coefs[2])/plane_coefs[1]
    u, v = getTransformMap(plane_coefs, disp, Q, ROIXZ, outputSize=[width, height])

    # if False:
    #     plt.figure()
    #     plt.plot(x.ravel()/1000, z.ravel()/1000, '.')
    #     plt.xlabel('x [m]')
    #     plt.ylabel('z [m]')
    #     plt.figure()
    #     plt.plot(u.ravel(), v.ravel(), '.')
    #     plt.xlabel('u [pixel]')
    #     plt.ylabel('v [pixel]')
    #     plt.show()

    bird_view = cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC)
    bird_view_mask = cv2.remap(np.uint8(mask), u.astype(np.float32), v.astype(np.float32), cv2.INTER_NEAREST)

    
    # overlay detected water
    overlay = bird_view.copy()
    for xx, zz in zip(mask_xyz[:, 0]/1000, mask_xyz[:, 2]/1000):
        try:
            x_int = int((xx - ROIXZ[0])/(ROIXZ[1] - ROIXZ[0]) * width)
            z_int = height - int((zz - ROIXZ[2])/(ROIXZ[3] - ROIXZ[2]) * height)
            cv2.circle(overlay, (x_int, z_int), radius=2, color=(255,0,0))
        except:
            pass

    return xyz, mask_xyz, bird_view, overlay, bird_view_mask

def track3D_mask(mask, plane_coefs, disp, Q, ROIXZ=[-10., 10., 0.5, 20.]):

    width = 1280
    height = 720
    #width = 647//2
    #height = 189
    u, v = getTransformMap(plane_coefs, disp, Q, ROIXZ, outputSize=[width, height])


    #bird_view = cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC)
    bird_view_mask = cv2.remap(np.uint8(mask), u.astype(np.float32), v.astype(np.float32), cv2.INTER_NEAREST)

    

    return bird_view_mask


def track3D_img(image, plane_coefs, disp, Q, ROIXZ=[-10., 10., 0.5, 20.], showAxes=False, show=False):

    width = 1280
    height = 720
    #width = 647//2
    #height = 189
    u, v = getTransformMap(plane_coefs, disp, Q, ROIXZ, outputSize=[width, height])


    bird_view = cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC)
    #bird_view_mask = cv2.remap(np.uint8(mask), u.astype(np.float32), v.astype(np.float32), cv2.INTER_NEAREST)

    

    return bird_view

def getReflectionAngle(disparity, plane_coefs, focal_length, u0=None, v0=None):
    '''Compute reflection angle from ground plane.
    Assuming ground plane coincide with the plane of water surface.

    Input:
        disp: stereo disparity map in pixels
        coefs: plane coefficients of ground plane disparity
        focal_length: camera focal length in pixels
        u0, v0: optical centre in pixels
    Output:
        alpha: reflection angle in radians

    Ref: refer to Nguyen et al ICRA 2017 paper for details of the formulas.
    '''
    # ground plane disparity
    u, v = np.meshgrid(np.arange(disparity.shape[1]),
                       np.arange(disparity.shape[0]))
    d = plane_coefs[0]*u + plane_coefs[1]*v + plane_coefs[2]

    # horizon line
    uHor, vHor = getHorizonLine(plane_coefs, disparity.shape[1])

    # refer following equation to camera & scene model in the paper.
    # uR and vR are u and v
    uR = u
    vR = v
    dR = d
    if u0 is None:
        uC = disparity.shape[1]//2
    else:
        uC = u0
    if v0 is None:
        vC = disparity.shape[0]//2
    else:
        vC = v0
    vI1 = vHor[uC]
    vI3 = vHor[uR]

    gamma = np.arctan2(vHor[-1] - vHor[0], uHor[-1] - uHor[0])
    cosGamma = np.cos(gamma)
    RI4 = (vR - vI3) * cosGamma
    OR = np.sqrt(focal_length**2 + (uC - uR)**2 + (vC - vR)**2)
    OI4 = np.sqrt(focal_length**2 + (vC - vI1)**2*cosGamma**2 +
                  (uC - uR)**2/cosGamma**2)
    # reflection angle
    alpha = np.arccos((OR**2 + OI4**2 - RI4**2)/(2*OR*OI4))

    return alpha

#def depth_validation(image, disparity, plane_coefs, focal_length,
#                     u0=None, v0=None, show=False):
#    '''Validate disparity of objects and its reflection on water
#
#    Input:
#        image: reference image
#        diparity: stereo disparity in pixels
#        plane_coeffs: plane coefficients of ground plane disparity
#        u0, v0: optical center in pixels
#        show: flag whether to display the result or not
#
#    Output:
#        prob: probability of a pixel to belong to a reflection
#    '''
#    # ground plane disparity
#    u, v = np.meshgrid(np.arange(disparity.shape[1]),
#                       np.arange(disparity.shape[0]))
#    d = plane_coefs[0]*u + plane_coefs[1]*v + plane_coefs[2]
#
#    # horizon line
#    uHor, vHor = getHorizonLine(plane_coefs, disparity.shape[1])
#
#    # refer following equation to camera & scene model in the paper.
#    # uR and vR are u and v
#    uR = u
#    vR = v
#    dR = d
#    if u0 is None:
#        uC = disparity.shape[1]//2
#    else:
#        uC = u0
#    if v0 is None:
#        vC = disparity.shape[0]//2
#    else:
#        vC = v0
#    vI1 = vHor[uC]
#    vI3 = vHor[uR]
#
#    gamma = np.arctan2(vHor[-1] - vHor[0], uHor[-1] - uHor[0])
#    cosGamma = np.cos(gamma)
#    RI4 = (vR - vI3) * cosGamma
#    OR = np.sqrt(focal_length**2 + (uC - uR)**2 + (vC - vR)**2)
#    OI4 = np.sqrt(focal_length**2 + (vC - vI1)**2*cosGamma**2 +
#                  (uC - uR)**2/cosGamma**2)
#    # reflection angle
#    alpha = np.arccos((OR**2 + OI4**2 - RI4**2)/(2*OR*OI4))
#
#    CI2 = (vC - vI1)*cosGamma
##    beta4 = np.arctan2(CI2, OI4)
##    I4J4 = CI2*(1 + OI4**2/CI2**2)
##    # d = d_R, disparity = d_S" or d_S
##    GR = (d - disparity) / (d/ (IJ - RI4) + disparity/RI4)
#
#    # simplified case where beta ~ 0
#    GR = (d - disparity) / disparity * RI4
#    IS = RT4 - 2.0 * GR
#    uS =


def demoTrack3D():
#    Folder = 'video_21-08-2016_14-21-39'
    Folder = 'video_21-08-2016_12-14-49'
    IndexRange = range(5000)

    map1x, map1y, map2x, map2y, mat, Q = \
        zedutils.getTransformFromConfig(FileName='SN1994.conf', Type='CAM_HD')

    for i in [180]: #[1760]: #[1080]: #[1340]: #IndexRange: #
        left_filename = os.path.join(Folder, 'test_new', 'frame_left_%04d.png' % i)
        mask_filename = os.path.join(Folder, 'test_new', 'mask_left_%04d.png' % i)
#        mask_predict_filename = os.path.join(Folder, 'test_new', 'predicted_mask_left_%04d.png' % i)
        plane_coefs_filename = os.path.join(Folder, 'test_new', 'plane_coefs_%04d.npy' % i)
        if os.path.exists(mask_filename):
            print('Found %s' % mask_filename)
            left_image = cv2.imread(left_filename)
#            mask_predict = cv2.imread(mask_predict_filename)
            mask = cv2.imread(mask_filename, 0)
            plane_coefs = np.load(plane_coefs_filename)
            xyz, mask_xyz, bird_view, overlay = track3D(left_image, mask, plane_coefs, Q, show=True)

        else:
            print('Missing %s' % mask_filename)

def demoPlaneFitting():
    d = np.load('disparity.npy') #[infHeight:, :]
    u, v = np.meshgrid(np.arange(d.shape[1]), np.arange(d.shape[0]))
    coefs, mask = fitPlane(u, v, d, infHeight=400) # =0)
    showPlane(u, v, d, coefs)
    plt.show()

if __name__ == '__main__':
#    demoPlaneFitting()
#    demoTrack3D()
    demoWarp3D()
