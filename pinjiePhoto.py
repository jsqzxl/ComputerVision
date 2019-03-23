#!/usr/bin/python3
#-*-coding:UTF-8-*-

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    top, bot, left, right = 100, 100, 0, 1000    #设定最后拼接完成的图像的大小
    filepath = 'C:\\Users\\jason_NK\\Desktop\\Program\\python\\Compution vision homework\\third_homework\\run_file\\'
    im1 = cv.imread(filepath + 'HW3Pic1.jpg')
    im2 = cv.imread(filepath + 'HW3Pic2.jpg')
    srcImg = cv.copyMakeBorder(im1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv.copyMakeBorder(im2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    im1_gray = cv.cvtColor(srcImg, cv.COLOR_BGR2GRAY)        #将图像转换为灰度图
    im2_gray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
    hessian = 600
    surf = cv.xfeatures2d.SURF_create(hessian) #海塞矩阵阈值越大，被检测到的特征点越少
    # 用surf算法检测关键点
    kp1, des1 = surf.detectAndCompute(im1_gray, None)
    kp2, des2 = surf.detectAndCompute(im2_gray, None)

    # 配置FLANNBASED算法的参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []
    pts1 = []
    pts2 = []
    # 筛选一些匹配比较好的点
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    #在图像上画出特征点并将匹配上的点连线
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv.drawMatchesKnn(im1_gray, kp1, im2_gray, kp2, matches, None, **draw_params)
    plt.figure(num='stitch', figsize=(8, 8))
    plt.subplot(2,1,1)
    plt.imshow(img3)

    rows, cols = srcImg.shape[:2]
    MIN_MATCH_COUNT = 10     #图片可拼接的最少特征点的数目
    #图像拼接
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        warpImg = cv.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]), flags=cv.WARP_INVERSE_MAP)
        print(warpImg.shape)
        for col in range(0, cols):
            if srcImg[:, col].any() and warpImg[:, col].any():
                left = col
                break
        for col in range(cols-1, 0, -1):
            if srcImg[:, col].any() and warpImg[:, col].any():
                right = col
                break

        res = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not srcImg[row, col].any():
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = srcImg[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(srcImg[row, col] * (1-alpha) + warpImg[row, col] * alpha, 0, 255)

        res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
        # 显示拼接结果
        plt.subplot(2,1,2)
        plt.imshow(res)
        plt.show()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None