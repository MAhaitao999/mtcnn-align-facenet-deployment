#########################################################################
# File Name: keras_onnx.py
# Author: Henry Ma
# mail: iloveicRazavi@gmail.com
# Create Time: 2021年02月04日 星期四 17时29分34秒
#########################################################################

# !/usr/bin/python
# -*- coding:utf-8 -*-

from builtins import *

import cv2
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input, MaxPool2D,
                                  Permute, Reshape)
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential

import keras
import keras2onnx
import onnx


#-----------------------------#
#    粗略获取人脸框
#    输出bbox位置和是否有人脸
#-----------------------------#
def create_Pnet(weight_path):
    inputs = Input(shape=[None, None, 3], name='input_1')

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 无激活函数, 线性
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)

    return model


if __name__ == '__main__':
    pnet_weights = 'model_data/pnet.h5'
    pnet = create_Pnet(pnet_weights)
    pnet_onnx_model = keras2onnx.convert_keras(pnet, pnet.name)
    pnet_onnx_name = 'model_data/pnet.onnx'
    onnx.save_model(pnet_onnx_model, pnet_onnx_name)

