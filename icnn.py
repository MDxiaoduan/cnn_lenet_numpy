# ------------------------------------------------------------------------ #
# cnn--Lenet                                                               #
# Reference:ImageNet Classification with Deep Convolutional Neura          #
# python 3.5                                                               #
# numpy                                                                    #
# 12/8/2017                                                                #
# UESTC--Duan                                                              #
# Data: CIFAR10   or  mnist                                                #
# ------------------------------------------------------------------------ #
# import Lib
from __future__ import division        # for 1/2=0.5
import numpy as np
import math
import cv2

import load_data

from ifunction import *
# convolve, sigmoid, ReLu, sigmoid_derivative, ReLu_derivative, Batch_Normalization, pooling, expand, deconvolution
"""
Lenet_model:
   input: 28*28
   convolution1:
         weight_1: size: [1, 6, 5, 5] （1个输入，6个输出，一共1*6个卷积核,每个卷积核大小5*5）
         bias_1: size: 6  6个输出所以只有6个bias(每个bias是一个数)
         output_size: 6*24*24   24=(28-5)/1+1
         pooling_1: size: [2, 2]  2*2大小做一个均值池化
         output_size: 6*12*12
   convolution2:
         weight_2: size: [6, 12, 5, 5] （6个输入，12个输出，一共6*12个卷积核,每个卷积核大小5*5）
         bias_2: size: 12
         output_size: 12*8*8   8=(12-5)/1+1
         pooling_2: size: [2, 2]  2*2大小做一个均值池化
         output_size: 12*4*4
   fully connected:
         input_size:12*4*4=192  --12个4*4输出变为一个向量
         weight_out: size: [192, 10]  (192个输入（每个输入就是一个数，10个分类输出）)
         bias_out: size: 10  10个输出
         output_size: 10
"""
class cnn_model:
    def __init__(self):
        """
        权重初始化方法：
            激活函数为sigmoid:
                 w = np.random.randn(n)/sqrt(n)        # n=k*k*c   k是输入大小，c是输入channel
            激活函数为ReLu:
                 w = np.random.randn(n)*sqrt(2.0/n)   # [Ref:https://arxiv.org/pdf/1502.01852.pdf]
            其他：
                w = np.random.randn(n)*sqrt(2.0/(n_in_size + n_out_size))
        """
        self.input_size = [28, 28]
        self.model_size = [1, 6, 12, 192, 10]     # [输入图片, 第一层卷积输出维数， 第二层卷积维数，  全链接输入， 分类标签数]
        self.weight_1 = np.random.randn(1, 6, 5, 5) * np.sqrt(2.0 / (1 + 6))
        self.bias_1 = np.random.randn(6)
        self.weight_2 = np.random.randn(6, 12, 5, 5) * np.sqrt(2.0 / (1 + 6))
        self.bias_2 = np.zeros(12)
        self.weight_out = np.random.randn(192, 10)
        self.bias_out = np.zeros(10)
        # self.batch_size = 1      # 多少张图片一起训练
        self.class_number = 10
        self.learning_rate = 1   # 学习速率
        self.iteration_number = 15000
        # 定义矩阵存储中间卷积后矩阵
        self.con1_out = np.zeros((self.model_size[1], 24, 24))
        self.pooling1_out = np.zeros((self.model_size[1], 12, 12))
        self.pooling2_out = np.zeros((self.model_size[2], 4, 4))
        self.full_con = np.zeros(192)
        self.t = np.zeros(self.class_number)
        self.d_out_w = np.zeros((192, 10))
        self.d_out_bias = np.zeros(10)
        self.der_full_con = np.zeros(192)
        self.der_pool_2 = np.zeros((12, 4, 4))
        self.der_con_2 = np.zeros((12, 8, 8))
        self.der_weight_2 = np.zeros((6, 12, 5, 5))
        self.d_con2_weight = np.zeros((6, 12, 5, 5))
        self.d_con2_bias = np.zeros(12)
        self.der_con_1 = np.zeros((6, 24, 24))
        self.der_weight_1 = np.zeros((1, 6, 5, 5))
        self.d_con1_weight = np.zeros((1, 6, 5, 5))
        self.d_con1_bias = np.zeros(6)
        self.der_out = np.zeros(10)
        self.con2_out = np.zeros((self.model_size[2], 8, 8))

    def cnn_forward(self, data):
        data = np.reshape(data, (self.input_size[0], self.input_size[1]))
        # convolution_1:
        for jj in range(self.model_size[0]):  # 1
            for kk in range(self.model_size[1]):  # 6
                self.con1_out[kk, :, :] = convolve(data, self.weight_1[jj, kk, :, :])  # 卷积
                self.con1_out[kk, :, :] = sigmoid(self.con1_out[kk, :, :] + self.bias_1[kk])  # sigmoid(Wx+b)
                # Batch_Normalization  con--batch_norm--ReLu
                # pooling_1:
                self.pooling1_out[kk, :, :] = pooling(self.con1_out[kk, :, :], [2, 2], 2)/4
        # convolution_2:   input:self.pooling1_out  [batch_size, 6, 12, 12]
        t = np.zeros((self.model_size[2], 8, 8))
        for kk in range(self.model_size[2]):  # 12
            for jj in range(self.model_size[1]):  # 6
                 t += convolve(self.pooling1_out[jj, :, :], self.weight_2[jj, kk, :, :])
            # ss = t/12
            self.con2_out[kk, :, :] = sigmoid(t[kk, :, :] + self.bias_2[kk])
            self.pooling2_out[kk, :, :] = pooling(self.con2_out[kk, :, :], [2, 2], 2)/4
        # fully connected:
        self.full_con = np.reshape(self.pooling2_out, 192)  # 先一个4*4按行reshape为向量，在12个维度
        for kk in range(self.model_size[4]):
            # for jj in range(self.model_size[3]):
            self.t[kk] = sigmoid((self.full_con * self.weight_out[:, kk]).sum() + self.bias_out[kk])
        return self.t, data

    def cnn_sigmoid(self, data, label):
        self.der_pool_1 = np.zeros((6, 12, 12))
        self.t, data = self.cnn_forward(data)
        # 梯度下降更新
        # for ii in range(self.batch_size):
        self.der_out = (self.t - label)  # *self.t*(1-self.t)
        print(self.t, label, self.der_out)
        self.d_out_bias = self.der_out
        self.d_out_w = np.dot(self.full_con[:, None], self.der_out[None, :])
        self.der_full_con = self.full_con*(np.reshape(np.dot(self.weight_out, self.der_out[:, None]), 192))   # check here
        self.der_pool_2 = np.reshape(self.der_full_con, (12, 4, 4))  # 直接reshape就可以 顺序和正向刚好一样 哈哈
        for ii in range(self.model_size[2]):
            self.der_con_2[ii, :, :] = expand(self.der_pool_2[ii, :, :], 2)  # [12, 8, 8]
        self.der_con_2 = self.der_con_2*self.con2_out*(1-self.con2_out)
        for ii in range(self.model_size[1]):
            for jj in range(self.model_size[2]):
                self.der_pool_1[ii, :, :] += deconvolution(self.der_con_2[jj, :, :], self.weight_2[ii, jj, :, :])
        for ii in range(self.model_size[1]):
            for jj in range(self.model_size[2]):
                self.d_con2_weight[ii, jj, :, :] = convolve(self.pooling1_out[ii, :, :], self.der_con_2[jj, :, :])
        for ii in range(self.model_size[2]):
            self.d_con2_bias[ii] = self.der_con_2[ii, :, :].sum()/64
        for ii in range(self.model_size[1]):
            self.der_con_1[ii, :, :] = expand(self.der_pool_1[ii, :, :], 2)
        self.der_con_1 = self.der_con_1*self.con1_out*(1-self.con1_out)
        for ii in range(self.model_size[1]):
            self.d_con1_weight[0, ii, :, :] = convolve(data, self.der_con_1[ii, :, :])
            self.d_con1_bias[ii] = self.der_con_1[ii, :, :].sum()/(24*24)
        # update
        self.weight_1 -= self.learning_rate*self.d_con1_weight
        self.bias_1 -= self.learning_rate*self.d_con1_bias
        self.weight_2 -= self.learning_rate * self.d_con2_weight
        self.bias_2 -= self.learning_rate * self.d_con2_bias
        self.weight_out -= self.learning_rate * self.d_out_w
        self.bias_out -= self.learning_rate * self.d_out_bias


if __name__ == "__main__":
    model = cnn_model()
    # load data
    get_data = load_data.mnist(1)
    for kk in range(model.iteration_number):
        print("iter:", kk)
        data, label = get_data.get_mini_bath("train")
        model.cnn_sigmoid(data, np.reshape(label, 10))
    acc = 0
    for jj in range(100):
        data, label = get_data.get_mini_bath("test")
        t, _ = model.cnn_forward(data)
        if np.argmax(t) == np.argmax(label):
            acc += 1
        else:
            continue
    print("Lenet Accuracy: ", acc/100)

# acc 0.94

