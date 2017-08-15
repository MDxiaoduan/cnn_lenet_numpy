from __future__ import division        # for 1/2=0.5
import numpy as np


def Batch_Normalization(batch_data):
    mean = np.mean(batch_data)
    std = np.std(batch_data)
    batch_norm_data = (batch_data - mean) / std
    return batch_norm_data


def convolve(image, kernel):  # image, kernel 都是二维矩阵(不一定是正方形)  stride = 1
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    output = np.zeros((iH-kH+1, iW-kW+1), dtype="float32")
    for y in np.arange(0, iH-kH+1):
        for x in np.arange(0, iW-kW+1):
            roi = image[y:y+kH, x:x+kW]
            k = (np.array(roi)*np.array(kernel)).sum()
            output[y, x] = k
    return output


def sigmoid(x):
    return 1./(1+np.exp(-x))


def ReLu(x):
    if x < 0:
        return 0
    else:
        return x


def sigmoid_derivative(x):     # sigmoid(x)的导数
    return (1./(1+np.exp(-x)))*(1-(1./(1+np.exp(-x))))


def ReLu_derivative(x):        # ReLu(x)的导数
    if x < 0:
        return 0
    else:
        return 1


def pooling(image, p_size, stride):     # 最好是简单的可以整除
    (iH, iW) = image.shape[:2]
    (kH, kW) = p_size
    out = np.zeros((int(iH/stride), int(iW/stride)))     # 输入输出的大小关系
    for ii in range(0, iH, stride):
        for jj in range(0, iW, stride):   # 相当于和np.ones((2,2))做卷积
            out[int(ii/stride), int(jj/stride)] = image[ii:(ii+kH), jj:(jj+kW)].sum()
    return out/4


def expand(input, stride):  # 二维矩阵
    (w, h) = input.shape
    out = np.zeros((w*stride, h*stride))
    for ii in range(0, w*stride, stride):
        for jj in range(0, h*stride, stride):
            out[ii:(ii + stride), jj:(jj+stride)] = input[int(ii/2), int(jj/2)]*np.ones((stride, stride))
    return out


def deconvolution(image, weight):    # 残差和卷积核得到上一层残差   in: [8, 8] [5, 5] out :[12, 12]
    (iH, iW) = image.shape[:2]
    (kH, kW) = weight.shape[:2]
    image_exp = np.zeros((iH+2*(kH-1), iW+2*(kW-1)))
    image_exp[4:12, 4:12] = image
    weight_transpose = np.zeros(kH*kW)
    weight_vec = np.reshape(weight, (1, -1))
    # for ii in range(kH*kW):
    #     weight_transpose[ii] = weight_vec[0, 24-ii]
    weight_transpose = np.reshape(weight_transpose, (kH, kW))
    out = convolve(image_exp, weight)
    return out
#
# x = np.random.randint(1, 2, (5, 5))
# y = np.random.randint(1, 2, (8, 8))
#
# print(deconvolution(y, x))





