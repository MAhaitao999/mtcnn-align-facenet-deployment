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
from functools import partial

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Activation, 
                          BatchNormalization, 
                          Concatenate, Conv2D, 
                          Dense, Dropout,
                          Flatten,
                          GlobalAveragePooling2D, 
                          Input,
                          Lambda,
                          MaxPool2D, MaxPooling2D,
                          Permute, 
                          Reshape,
                          add)
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


#------------- facenet 辅助函数 --------------#
def scaling(x, scale):
    return x * scale


def _generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))


def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                               scale=False, name=_generate_layer_name('BatchNorm', prefix=name), trainable=False)(x)

    if activation is not None:
        x = Activation(activation, name=_generate_layer_name('Activation', prefix=name))(x)
    return x


def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    channel_axis = 3
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))

    name_fmt = partial(_generate_layer_name, prefix=prefix)

    if block_type == 'Block35':
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'Block17':
        branch_0 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 128, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = conv2d_bn(branch_1, 128, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1]
    elif block_type == 'Block8':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 192, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = conv2d_bn(branch_1, 192, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1]

    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)
    up = conv2d_bn(mixed, K.int_shape(x)[channel_axis], 1, activation=None, use_bias=True,
                   name=name_fmt('Conv2d_1x1'))
    up = Lambda(scaling,
                output_shape=K.int_shape(up)[1:],
                arguments={'scale': scale})(up)
    x = add([x, up])
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)
    return x


def InceptionResNetV1(input_shape=(160, 160, 3),
                      classes=128,
                      dropout_keep_prob=0.8):
    channel_axis = 3
    inputs = Input(shape=input_shape, name='input_1')
    # 160, 160, 3 -> 77, 77, 64
    x = conv2d_bn(inputs, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
    x = conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
    x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')
    # 77, 77, 64 -> 38, 38, 64
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    
    # 38, 38, 64 -> 17, 17, 256
    x = conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
    x = conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
    x = conv2d_bn(x, 256, 3, strides=2, padding='valid', name='Conv2d_4b_3x3')

    # 5x Block35 (Inception-ResNet-A block):
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x, scale=0.17, block_type='Block35', block_idx=block_idx)
    
    # Reduction-A block:
    # 17, 17, 256 -> 8, 8, 896
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a')
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 192, 3, name=name_fmt('Conv2d_0b_3x3', 1))
    branch_1 = conv2d_bn(branch_1, 256, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 1))
    branch_pool = MaxPooling2D(3, strides=2, padding='valid', name=name_fmt('MaxPool_1a_3x3', 2))(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

    # 10x Block17 (Inception-ResNet-B block)
    for block_idx in range(1, 11):
        x = _inception_resnet_block(x,
                                    scale=0.1,
                                    block_type='Block17',
                                    block_idx=block_idx)

    # Reduction-B block
    # 8, 8, 896 -> 3, 3, 1792
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')
    branch_0 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 256, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 1))
    branch_2 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 256, 3, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2, 256, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 2))
    branch_pool = MaxPooling2D(3, strides=2, padding='valid', name=name_fmt('MaxPool_1a_3x3', 3))(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

    # 5x Block8 (Inception-ResNet-C block):
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x,
                                    scale=0.2,
                                    block_type='Block8',
                                    block_idx=block_idx)
    x = _inception_resnet_block(x, scale=1., activation=None, block_type='Block8', block_idx=6)

    # 平均池化
    x = GlobalAveragePooling2D(name='AvgPool')(x)
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)
    # 全连接层到128
    x = Dense(classes, use_bias=False, name='Bottleneck')(x)
    bn_name = _generate_layer_name('BatchNorm', prefix='Bottleneck')
    feature = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False,
                                 name=bn_name, trainable=False)(x)

    # 创建模型
    model = Model([inputs], [feature], name='inception_resnet_v1')

    return model


def create_Facenet(weight_path):
    model = InceptionResNetV1()
    # print(model.summary())
    model.load_weights(weight_path, by_name=True)

    return model


if __name__ == '__main__':
    pnet_weights = 'model_data/pnet.h5'
    pnet = create_Pnet(pnet_weights)
    pnet.save('model_data/PNET.h5')
    pnet_onnx_model = keras2onnx.convert_keras(pnet, pnet.name)
    pnet_onnx_name = 'model_data/pnet.onnx'
    onnx.save_model(pnet_onnx_model, pnet_onnx_name)

    rnet_weights = 'model_data/rnet.h5'
    rnet = create_Rnet(rnet_weights)
    rnet.save('model_data/RNET.h5')
    rnet_onnx_model = keras2onnx.convert_keras(rnet, rnet.name)
    rnet_onnx_name = 'model_data/rnet.onnx'
    onnx.save_model(rnet_onnx_model, rnet_onnx_name)
    
    onet_weights = 'model_data/onet.h5'
    onet = create_Onet(onet_weights)
    onet.save('model_data/ONET.h5')
    onet_onnx_model = keras2onnx.convert_keras(onet, onet.name)
    onet_onnx_name = 'model_data/onet.onnx'
    onnx.save_model(onet_onnx_model, onet_onnx_name)

    facenet_weights = 'model_data/facenet.h5'
    facenet = create_Facenet(facenet_weights)
    facenet.save('model_data/FACENET.h5')
    facenet_onnx_model = keras2onnx.convert_keras(facenet, facenet.name)
    facenet_onnx_name = 'model_data/facenet.onnx'
    onnx.save_model(facenet_onnx_model, facenet_onnx_name)
