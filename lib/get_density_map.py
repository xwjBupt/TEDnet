import cv2
import math
import numpy as np
import scipy.ndimage


def get_fix_gaussian(im, points):
    '''

    :param im: original image
    :param points: gt,n*2,represent a ground truth position
    :return: a image-size density map in which is every ground truth points position appled with gaussian filter
    '''

    rgb = False
    if len(im.shape)==3:
        h, w, c = im.shape
        rgb = True
        im = im[:,:,0]
    else:
        h,w = im.shape

    im_density = np.zeros_like(im, dtype=np.float64)

    if points is None:
        return im_density
    if points.shape[0] == 1:
        x1 = max(0, min(w-1, round(points[0, 0])))
        y1 = max(0, min(h-1, round(points[0, 1])))
        im_density[y1, x1] = 255
        return im_density
    for j in range(points.shape[0]):
        f_sz = 15
        sigma = 4.0
        H = np.multiply(cv2.getGaussianKernel(f_sz, sigma), (cv2.getGaussianKernel(f_sz, sigma)).T)
        x = min(w-1, max(0, abs(int(math.floor(points[j, 0])))))
        y = min(h-1, max(0, abs(int(math.floor(points[j, 1])))))
        if x >= w or y >= h:
            continue
        x1 = x - f_sz//2 + 0
        y1 = y - f_sz//2 + 0
        x2 = x + f_sz//2 + 1
        y2 = y + f_sz//2 + 1
        dfx1, dfy1, dfx2, dfy2 = 0, 0, 0, 0
        change_H = False

        #deal with the boundary
        if x1 < 0:
            dfx1 = abs(x1) + 0
            x1 = 0
            change_H = True
        if y1 < 0:
            dfy1 = abs(y1) + 0
            y1 = 0
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
        x1h, y1h, x2h, y2h = 1 + dfx1, 1 + dfy1, f_sz - dfx2, f_sz - dfy2
        if change_H is True:
            H = np.multiply(cv2.getGaussianKernel(y2h-y1h+1, sigma), (cv2.getGaussianKernel(x2h-x1h+1, sigma)).T)
        im_density[y1:y2, x1:x2] += H
    if rgb:
        temp = np.ones((c,im_density.shape[0],im_density.shape[1]))
        im_density = temp * im_density
        im_density = im_density.transpose([1,2,0])

    return im_density



def get_geo_gaussian(im,points,beta=0.3,knn = 4):
    '''
    这个方法是找某个人头附近的最近K邻的距离之和的beta倍做为当前此点的高斯核大小
    :param gt:
    :return:density map
    '''
    gt = np.zeros_like(im, dtype=np.float64)
    for i in range(0, len(points)):
        if int(points[i][1]) < im.shape[0] and int(points[i][0]) < im.shape[1]:
            gt[int(points[i][1]), int(points[i][0]),:] = 1
    # print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=knn)
    #
    # print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*beta
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density
    # print('done.')