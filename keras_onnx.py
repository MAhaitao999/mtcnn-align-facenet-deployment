#########################################################################
# File Name: keras_onnx.py
# Author: Henry Ma
# mail: iloveicRazavi@gmail.com
# Create Time: 2021年02月04日 星期四 17时29分34秒
#########################################################################

# !/usr/bin/python
# -*- coding:utf-8 -*-

import os
os.environ['TF_KERAS'] = '0'
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


#-----------------------------#
#    mtcnn的第二段
#    精修框
#-----------------------------#
def create_Rnet(weight_path):
    inputs = Input(shape=[24, 24, 3], name='input_1')

    # 24, 24, 3 -> 22, 22, 28 -> 11, 11, 28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 11, 11, 28 -> 9, 9, 48 -> 4, 4, 48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4, 4, 48 -> 3, 3, 64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 3, 3, 64 -> 64, 3, 3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    # 128 -> 2
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    # 128 -> 4
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)

    return model


#-----------------------------#
#    mtcnn的第三段
#    精修框并获得五个点
#-----------------------------#
def create_Onet(weight_path):
    inputs = Input(shape=[48, 48, 3], name='input_1')
    
    # 48, 48, 3 -> 46, 46, 32 -> 23, 23, 32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 23, 23, 32 -> 21, 21, 64 -> 10, 10, 64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 8, 8, 64 -> 4, 4, 64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 4, 4, 64 -> 3, 3, 128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # 3, 3, 128 -> 128, 3, 3 -> 1152
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 1152 -> 256
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 256 -> 2
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    # 256 -> 4
    bbox_regress = Dense(4, name='conv6-2')(x)
    # 256 -> 10
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([inputs], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)
    
    return model


if __name__ == '__main__':
    pnet_weights = 'model_data/pnet.h5'
    pnet = create_Pnet(pnet_weights)
    pnet_onnx_model = keras2onnx.convert_keras(pnet, pnet.name)
    pnet_onnx_name = 'model_data/pnet.onnx'
    onnx.save_model(pnet_onnx_model, pnet_onnx_name)

    rnet_weights = 'model_data/rnet.h5'
    rnet = create_Rnet(rnet_weights)
    rnet_onnx_model = keras2onnx.convert_keras(rnet, rnet.name)
    rnet_onnx_name = 'model_data/rnet.onnx'
    onnx.save_model(rnet_onnx_model, rnet_onnx_name)
    
    onet_weights = 'model_data/onet.h5'
    onet = create_Onet(onet_weights)
    onet_onnx_model = keras2onnx.convert_keras(onet, onet.name)
    onet_onnx_name = 'model_data/onet.onnx'
    onnx.save_model(onet_onnx_model, onet_onnx_name)
