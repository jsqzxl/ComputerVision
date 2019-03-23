#!/usr/bin/python3
#-*-coding:UTF-8-*-
#author:jisenquan
#date:2012-11-17
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 将彩色图像转换为灰度图像
# 转换公式：Y = 0.299R + 0.587G + 0.114B
#           U = -0.147R - 0.289G + 0.436B
#           V = 0.615R - 0.515G - 0.1B
def color_rgb2gray(src):
    gray = np.empty([src.shape[0],src.shape[1]])
    U = np.empty([src.shape[0], src.shape[1]])
    V = np.empty([src.shape[0], src.shape[1]])
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            gray[i,j] = 0.299 * src[i,j,0] + 0.587 * src[i,j,1] + 0.114 * src[i,j,2]
            U[i,j] = -0.147 * src[i,j,0] - 0.289 * src[i,j,1] + 0.436 * src[i,j,2]
            V[i,j] = 0.615 * src[i,j,0] - 0.515 * src[i,j,1] - 0.1 * src[i,j,2]
    return gray,U,V

#绘制灰度直方图
def histogram_image(gray):
    histogram_array = [0 for _ in range(256)]
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            histogram_array[int(gray[i,j])] = histogram_array[int(gray[i,j])] + 1
    return histogram_array

#因为YUV转RGB以后有的数据不在[0-255]之间，故需要归一化处理
def normalize_data(dis_img):
    dis_img[:, :, 0] = ((dis_img[:, :, 0] - dis_img[:, :, 0].min()) / (dis_img[:, :, 0].max() - dis_img[:, :, 0].min())) * 255
    dis_img[:, :, 1] = ((dis_img[:, :, 1] - dis_img[:, :, 1].min()) / (dis_img[:, :, 1].max() - dis_img[:, :, 1].min())) * 255
    dis_img[:, :, 2] = ((dis_img[:, :, 2] - dis_img[:, :, 2].min()) / (dis_img[:, :, 2].max() - dis_img[:, :, 2].min())) * 255
    return dis_img

#获取均衡化后的灰度图
#计算公式：dis = p * 255 ;p是一个累积概率
def get_disgray_disimg(histogram_array,gray,src_u,src_v):
    sk = [0.0 for _ in range(256)]
    dis_gray = np.empty([gray.shape[0],gray.shape[1]])
    dis_img = np.empty([img.shape[0],img.shape[1],img.shape[2]])
    num = gray.shape[0] * gray.shape[1]
    for i in range(len(histogram_array)):
        if i == 0:
            sk[i] = histogram_array[i] / num
        else:
            sk[i] = sk[i-1] + histogram_array[i] / num
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            dis_gray[i,j] = sk[int(gray[i,j])] * 255
            dis_img[i,j,0] = dis_gray[i,j] + 1.14 * src_v[i,j]
            dis_img[i,j,1] = dis_gray[i,j] - 0.39 * src_u[i,j] - 0.58 * src_v[i,j]
            dis_img[i,j,2] = dis_gray[i,j] + 2.03 * src_u[i,j]

    return dis_gray,dis_img

filepath = 'C:\\Users\\jason_NK\\Desktop\\Program\\python\\Compution vision homework\\second\\zhifangtu_balance\\HW2Pic1.png'
img = cv.imread(filepath)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
# 以下被注释掉的程序段是调用库函数完成图像的直方图均衡化
'''
imYUV = cv.cvtColor(img,cv.COLOR_BGR2YUV)
imYUV[:,:,0] = cv.equalizeHist(imYUV[:,:,0])
oytput = cv.cvtColor(imYUV,cv.COLOR_YUV2BGR)
cv.imshow('src',img)
cv.imshow('dis',img)
if cv.waitKey() & 0xFF == 27:
    cv.destroyAllWindows()
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#plt.imshow(img)
'''

##以下程序段是自己写的直方图均衡化程序
gray,src_u,src_v = color_rgb2gray(img)

axis = [i for i in range(256)]  #直方图横轴坐标
histogram_array = histogram_image(gray)   #得到图像的亮度直方图
dis_gray,dis_img = get_disgray_disimg(histogram_array,gray,src_u,src_v) # 均衡化后的亮度图和彩色图
dis_histogram_array = histogram_image(dis_gray)  #均衡化后的直方图
#dis_img = normalize_data(dis_img)
dis_img = np.where(dis_img < 0,0,np.where(dis_img > 255,255,dis_img))  #将像素值限制在0-255

#将图片可视化
plt.figure(num = 'figure1',figsize=(12,12))
plt.subplot(3,2,1)
plt.title('original gray')
plt.imshow(gray)

plt.subplot(3,2,2)
plt.plot(axis,histogram_array)
plt.title('original histogram')

plt.subplot(3,2,3)
plt.imshow(dis_gray)
plt.title('dis_gray')

plt.subplot(3,2,4)
plt.plot(axis,dis_histogram_array)
plt.title('dis_histogram')

plt.subplot(3,2,5)
plt.imshow(img)
plt.title('original image')

plt.subplot(3,2,6)
plt.imshow(dis_img.astype(np.uint8))
plt.title('dis_image')
plt.show()