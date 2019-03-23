#!/usr/bin/python3
#-*-coding:UTF-8-*-
#author:jisenquan
#date:2018-11-17
#python3
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
import math
import os

#对三通道彩色图像添加高斯噪声
def gauss_noise(src,mu,std):
    noise_img = np.empty([src.shape[0],src.shape[1],img.shape[2]])
    src.astype('float')
    #for i in range(src.shape[0]):
    #    for j in range(src.shape[1]):
    #        noise_img[i][j][0] = src[i][j][0] + random.gauss(0,0.1)
    #        noise_img[i][j][1] = src[i][j][1] + random.gauss(0, 0.1)
    #        noise_img[i][j][2] = src[i][j][2] + random.gauss(0, 0.1)
    Gauss_noise = np.random.normal(mu,std,(img.shape[0],img.shape[1],img.shape[2]))
    noise_img = src + Gauss_noise
    noise_img = np.where(noise_img < 0,0,np.where(noise_img > 255,255,noise_img)) #将像素值限制在0-255
    noise_img = noise_img.astype(np.uint8)   #变换为uint
    return noise_img

# 计算得到高斯核
#size:核的大小，默认核为方阵，
#std:标准差
def gauss_kernel(size,std):
    kernel = np.empty([5,5])
    r = int(size/2)
    for i in range(size):
        for j in range(size):
            res1 = 1/(2*math.pi*std*std)
            res2 = math.exp(-((i-r)**2 + (j-2)**2)/(2 * std**2))
            kernel[i,j] = res1 * res2
    kernel = kernel / sum(sum(kernel))
    return kernel
#定义高斯卷积函数，在高斯滤波的过程中执行卷积操作
#kernel:卷积核
#data:参数卷积的数据块
def gauss_convolve(kernel,data):
    con_temp = np.multiply(kernel,data)
    con = sum(sum(con_temp))
    return con
#得到中值滤波的核
#size:核的大小
def media_kernel(size):
    kernel = np.ones([size,size])
    return kernel
#计算中值卷积，并返回中值
#kernel:卷积核
#data:待卷积的数据块
def media_convolve(kernel,data):
    con_temp = np.multiply(kernel,data)
    con = np.median(con_temp)
    return con
#中值滤波函数
#noise_img:噪声图片
#size:卷积核的大小
def media_filter(noise_img,size):
    kernel = media_kernel(size)
    media_filter_img = np.empty([noise_img.shape[0], noise_img.shape[1], noise_img.shape[2]])
    #对图片进行填充
    pad_noise_img_r = np.pad(noise_img[:, :, 0], ((2, 2), (2, 2)), 'constant', constant_values=(0, 0))
    pad_noise_img_g = np.pad(noise_img[:, :, 1], ((2, 2), (2, 2)), 'constant', constant_values=(0, 0))
    pad_noise_img_b = np.pad(noise_img[:, :, 2], ((2, 2), (2, 2)), 'constant', constant_values=(0, 0))

    for row in range(pad_noise_img_b.shape[0] - 4):
        for col in range(pad_noise_img_b.shape[1] - 4):
            media_filter_img[row, col, 0] = media_convolve(kernel, pad_noise_img_r[row:row + 5, col:col + 5])
            media_filter_img[row, col, 1] = media_convolve(kernel, pad_noise_img_g[row:row + 5, col:col + 5])
            media_filter_img[row, col, 2] = media_convolve(kernel, pad_noise_img_b[row:row + 5, col:col + 5])
    media_filter_img = media_filter_img.astype(np.uint8)  #将像素值转为uint8

    return media_filter_img

#高斯滤波函数
#noise_img:噪声图片
#size:核的大小
#sigma:高斯分布的标准差
#注：均值默认为0
def gauss_filter(noise_img,size,sigma):
    kernel = gauss_kernel(size, sigma)
    pad_size = int(size / 2)
    gauss_filter_img = np.empty([noise_img.shape[0],noise_img.shape[1],noise_img.shape[2]])
    pad_noise_img_r = np.pad(noise_img[:,:,0],((pad_size,pad_size),(pad_size,pad_size)),'constant',constant_values=(0,0))
    pad_noise_img_g = np.pad(noise_img[:, :,1], ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=(0, 0))
    pad_noise_img_b = np.pad(noise_img[:, :,2], ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=(0, 0))
    #print(pad_noise_img_b.shape[0],pad_noise_img_b.shape[1])

    for row in range(pad_noise_img_b.shape[0]-4):
        for col in range(pad_noise_img_b.shape[1]-4):
            gauss_filter_img[row,col,0] = gauss_convolve(kernel,pad_noise_img_r[row:row+5,col:col+5])
            gauss_filter_img[row,col,1] = gauss_convolve(kernel,pad_noise_img_g[row:row+5,col:col+5])
            gauss_filter_img[row,col,2] = gauss_convolve(kernel,pad_noise_img_b[row:row+5,col:col+5])
    gauss_filter_img = gauss_filter_img.astype(np.uint8)

    return gauss_filter_img
#双边滤波的空间距离核
#size:核的大小
#sigma_d:空间距离的标准差
def bilateral_kernel(size,sigma_d):
    wd = np.empty([size,size])
    r = int(size / 2)
    for i in range(size):
        for j in range(size):
            res1 = 1 / (2 * math.pi * sigma_d * sigma_d)
            res2 = math.exp(-((i - r) ** 2 + (j - 2) ** 2) / (2 * sigma_d ** 2))
            wd[i, j] = res1 * res2
    return wd
#双边滤波的卷积操作
#wd:空间距离权值矩阵
#data:待处理的数据块
#sigmar:像素核的标准差
def bilateral_convole(wd,data,sigmar):
    i = int(data.shape[0]/2)
    value = data[i,i]
    wr = np.exp(-np.power(data-value,2)/(2 * sigmar**2))
    w = np.multiply(wd,wr)
    s = np.multiply(data,w)
    con = sum(sum(s)) / sum(sum(w))
    return con
#双边滤波函数
#noise_img:噪音图片
#size:核的大小
#sigmad:空间核的标准差
#sigmar:像素核的标准差
def bilateral_filter(noise_img,size,sigmd,sigmar):
    w1 = bilateral_kernel(size, sigmd)
    pad_size = int(w1.shape[0] / 2)
    bilateral_filter_img = np.empty([noise_img.shape[0], noise_img.shape[1], noise_img.shape[2]])
    pad_noise_img_r = np.pad(noise_img[:, :, 0], ((pad_size, pad_size), (pad_size, pad_size)), 'constant',
                             constant_values=(0, 0))
    pad_noise_img_g = np.pad(noise_img[:, :, 1], ((pad_size, pad_size), (pad_size, pad_size)), 'constant',
                             constant_values=(0, 0))
    pad_noise_img_b = np.pad(noise_img[:, :, 2], ((pad_size, pad_size), (pad_size, pad_size)), 'constant',
                             constant_values=(0, 0))
    #print(pad_noise_img_b.shape[0], pad_noise_img_b.shape[1])

    for row in range(pad_noise_img_b.shape[0] - 4):
        for col in range(pad_noise_img_b.shape[1] - 4):
            bilateral_filter_img[row, col, 0] = bilateral_convole(w1, pad_noise_img_r[row:row + 5, col:col + 5],sigmar)
            bilateral_filter_img[row, col, 1] = bilateral_convole(w1, pad_noise_img_g[row:row + 5, col:col + 5],sigmar)
            bilateral_filter_img[row, col, 2] = bilateral_convole(w1, pad_noise_img_b[row:row + 5, col:col + 5],sigmar)
    bilateral_filter_img = bilateral_filter_img.astype(np.uint8)

    return bilateral_filter_img


if __name__ == '__main__':

    filepath = 'C:\\Users\\jason_NK\\Desktop\\Program\\python\\Compution vision homework\\second\\linear_filter\\HW2Pic2.png'
    mu = 0      #噪音的平均值
    std = 15    #噪音的标准差

    img = cv.imread(filepath)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)  #matplotlib模块默认通道顺序是RGB
    fig1 = plt.figure('original_img')
    plt.imshow(img)
    fig1.show()

    noise_img = gauss_noise(img,mu,std)  # 给原图添加高斯噪声
    #print(noise_img.shape[0],noise_img.shape[1])
    fig2 = plt.figure('noise_img')
    plt.imshow(noise_img)
    fig2.show()
    #os.system('pause')

    #双边滤波
    bilateral_filter_img = bilateral_filter(noise_img,5,2,15)
    fig3 = plt.figure('bilateral_filter_img')
    plt.imshow(bilateral_filter_img)
    fig3.show()

    #高斯滤波
    gauss_filter_img = gauss_filter(noise_img,5,15)
    fig4 = plt.figure('gauss_filter_img')
    plt.imshow(gauss_filter_img)
    fig4.show()

    #中值滤波
    media_filter_img = media_filter(noise_img,5)
    fig5 = plt.figure('media_filter_img')
    plt.imshow(media_filter_img)
    plt.show()

    #以下代码是库函数实现
    '''
    gauss_filter_img = cv.GaussianBlur(noise_img,(5,5),std)    #高斯滤波
    media_filter_img = cv.medianBlur(noise_img,5)              #中值滤波
    bilateral_filter_img = cv.bilateralFilter(noise_img,2,100,100) #双边滤波

    plt.figure(num='filter',figsize=(8,8))
    plt.subplot(2,3,1)
    plt.title('original image')
    plt.imshow(img)

    plt.subplot(2,3,2)
    plt.title('noise_img')
    plt.imshow(noise_img)

    plt.subplot(2,3,3)
    plt.title('gauss filter img')
    plt.imshow(gauss_filter_img)

    plt.subplot(2,3,4)
    plt.title('media filter img')
    plt.imshow(media_filter_img)

    plt.subplot(2,3,5)
    plt.title('bilateral filter img')
    plt.imshow(bilateral_filter_img)

    plt.show()
    '''
