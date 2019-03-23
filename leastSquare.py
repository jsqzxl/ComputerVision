#author:jisenquan
#data:2018-10-30
#导入相关的包
import numpy as np
from numpy import *
import pylab as pl
import argparse

#定义命令行参数：iter:迭代次数   path:储存直线参数的文件  lamda:学习率
ap = argparse.ArgumentParser()
ap.add_argument("-i","--iter",type=int,required=True,help="the number of iteration")
ap.add_argument("-p","--path",type=str,required=True,help="data file path")
ap.add_argument("-l","--lamda",type=float,default=0.1)

args = vars(ap.parse_args())

# 多项式函数
def fit_fun(p,lines):
    p1 = np.append(p,1)      #将点p的坐标变成其次坐标
    p1 = mat(p1)            #将数组变成矩阵，便于下一步计算
    p1 = p1.T
    f = lines*p1             #计算点p1到所有直线的距离
    return f

#代价函数
def cost_fun(p,lines):
    f = fit_fun(p,lines)
    D = f.T * f               #点p到所有直线距离的平方
    return D

#更新x,y
def update_p(p,lines):
    f = fit_fun(p,lines)
    px = f.T * lines[:,0]
    py = f.T * lines[:,1]
    new_x = p[0] - args["lamda"] * px[0,0]       #使用梯度下降法更新点p的横坐标
    new_y = p[1] - args["lamda"] * py[0,0]       #使用梯度下降法更新点p的纵坐标
    return new_x,new_y

if __name__ == '__main__':

    lines = mat([0,0,0])               #相当于定义了一个空矩阵
    #p_init = np.random.randn(2)        #用随机数初始化带求点
    p_init = np.array([0.5,0.5])
    f = open(args["path"],'r')        #读取直线数据
    line = f.readline()

    while line:
        a,b,c = line.strip('\n').split()
        lines = np.row_stack((lines,[int(a),int(b),int(c)]))
        line = f.readline()
    f.close()
    lines = delete(lines,0,axis=0)      #删除直线矩阵中的没用的第一行

    D1 = 0
    D2 = 0
    D3 = 0                               #D1，D2，D3相当于滤波，提前判断收敛条件
    count = 0
    D_record = [];

    while True:

        D = cost_fun(p_init,lines)
        D3 = D2
        D2 = D1
        D1 = D[0,0]
        D_record.append(D1)
        if (D3 == D2 == D1) or count == args["iter"]:
            break
        else:
            new_x,new_y = update_p(p_init,lines)
            p_init = array([new_x,new_y])
        count = count + 1

    print("D3:",D3)               #输出最后距离的平方
    print("p_init:",p_init)       #输出计算得到点的坐标
    print("count:",count)         #输出迭代次数
    counts = np.linspace(0,count,len(D_record))
    pl.plot(counts,D_record)
    pl.show()                          #将距离的下降过程画出来