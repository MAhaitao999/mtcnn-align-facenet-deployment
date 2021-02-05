#########################################################################
# File Name: mtcnn.py
# Author: Henry Ma
# mail: iloveicRazavi@gmail.com
# Create Time: 2021年02月05日 星期五 18时16分17秒
#########################################################################

# !/usr/bin/python
# -*- coding:utf-8 -*-

import time
from builtins import *

import cv2
import numpy as np

import tritonclient.http
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

import utils


class mtcnn(object):
    def __init__(self):
        self.triton_client = tritonclient.http.InferenceServerClient("127.0.0.1:8000", verbose=False)
        #-----------------------------#
        #        mtcnn的第一段
        #        粗略获取人脸框
        #        输出bbox位置和是否有人脸
        #-----------------------------#
        #-----------------------------#
        #        mtcnn的第二段
        #        精修框
        #-----------------------------#
        #-----------------------------#
        #        mtcnn的第三段
        #        精修框并获得五个点
        #-----------------------------#
        self.Pnet_inputs = ['input_1']
        self.Pnet_outputs = ['conv4-1', 'conv4-2']
        self.Rnet_inputs = ['input_1']
        self.Rnet_outputs = ['conv5-1', 'conv5-2']
        self.Onet_inputs = ['input_1']
        self.Onet_outputs = ['conv6-1', 'conv6-2', 'conv6-3']

    def detectFace(self, img, threshold):
        #-----------------------------#
        #        归一化
        #-----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        print("orgin image's shape is: ", origin_h, origin_w)
        #-----------------------------#
        #        计算原始输入图像
        #        每一次缩放的比例
        #-----------------------------#
        scales = utils.calculateScales(img)

        out = []

        #-----------------------------#
        #        粗略计算人脸框
        #        pnet部分
        #-----------------------------#
        for scale in scales:
            pnet_inputs = []
            pnet_outputs = []
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = np.expand_dims(scale_img, 0).astype(np.float32)
            
            pnet_inputs.append(tritonclient.http.InferInput(self.Pnet_inputs[0], inputs.shape, 'FP32'))
            pnet_inputs[0].set_data_from_numpy(inputs, binary_data=True)

            pnet_outputs.append(tritonclient.http.InferRequestedOutput(self.Pnet_outputs[0], binary_data=True))
            pnet_outputs.append(tritonclient.http.InferRequestedOutput(self.Pnet_outputs[1], binary_data=True))

            output = self.triton_client.infer("pnet", inputs=pnet_inputs, outputs=pnet_outputs)
            # print(output.as_numpy(self.Pnet_outputs[0]).shape)
            # print(output.as_numpy(self.Pnet_outputs[1]).shape)
            output = [output.as_numpy(self.Pnet_outputs[0])[0], output.as_numpy(self.Pnet_outputs[1])[0]]
            out.append(output)

            # print(out)

        rectangles = []
        #-------------------------------------------------#
        #   在这个地方我们对图像金字塔的预测结果进行循环
        #   取出每张图片的种类预测和回归预测结果
        #-------------------------------------------------#
        for i in range(len(scales)):
            #------------------------------------------------------------------#
            #   为了方便理解，这里和视频上看到的不太一样
            #   因为我们在上面对图像金字塔循环的时候就把batch_size维度给去掉了
            #------------------------------------------------------------------#
            cls_prob = out[i][0][:, :, 1]
            roi = out[i][1]
            #--------------------------------------------#
            #   取出每个缩放后图片的高宽
            #--------------------------------------------#
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            #--------------------------------------------#
            #   解码的过程
            #--------------------------------------------#
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        #-----------------------------------------#
        #    进行非极大抑制
        #-----------------------------------------#
        rectangles = np.array(utils.NMS(rectangles, 0.7))
        # print(rectangles)

        if len(rectangles) == 0:
            return rectangles

        #-----------------------------------------#
        #    稍微精确计算人脸框
        #    Rnet部分
        #-----------------------------------------#
        predict_24_batch = []
        for rectangle in rectangles:
            #--------------------------------------------#
            #    利用获取到的粗略坐标，在原图上进行截取
            #--------------------------------------------#
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #--------------------------------------------#
            #    将截取到的图片进行resize，调整成24x24的大小
            #--------------------------------------------#
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        # print(np.array(predict_24_batch).shape)

        rnet_inputs = []
        rnet_outputs = []
        rnet_inputs.append(tritonclient.http.InferInput(self.Rnet_inputs[0], np.array(predict_24_batch).shape, 'FP32'))
        rnet_inputs[0].set_data_from_numpy(np.array(predict_24_batch).astype(np.float32), binary_data=True)
        
        rnet_outputs.append(tritonclient.http.InferRequestedOutput(self.Rnet_outputs[0], binary_data=True))
        rnet_outputs.append(tritonclient.http.InferRequestedOutput(self.Rnet_outputs[1], binary_data=True))

        output = self.triton_client.infer("rnet", inputs=rnet_inputs, outputs=rnet_outputs)
        # print(output.as_numpy(self.Rnet_outputs[0]).shape)
        # print(output.as_numpy(self.Rnet_outputs[1]).shape)
        cls_prob, roi_prob = output.as_numpy(self.Rnet_outputs[0]), output.as_numpy(self.Rnet_outputs[1])
        # print('cls_prob is: ')
        # print(cls_prob)
        # print('roi_prob is: ')
        # print(roi_prob)
        #-------------------------------------#
        #   解码的过程
        #-------------------------------------#
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        # print(rectangles)

        #-----------------------------#
        #   计算人脸框
        #   onet部分
        #-----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            #------------------------------------------#
            #   利用获取到的粗略坐标，在原图上进行截取
            #------------------------------------------#
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #-----------------------------------------------#
            #   将截取到的图片进行resize，调整成48x48的大小
            #-----------------------------------------------#
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        onet_inputs = []
        onet_outputs = []
        onet_inputs.append(tritonclient.http.InferInput(self.Onet_inputs[0], np.array(predict_batch).shape, 'FP32'))
        onet_inputs[0].set_data_from_numpy(np.array(predict_batch).astype(np.float32), binary_data=True)

        onet_outputs.append(tritonclient.http.InferRequestedOutput(self.Onet_outputs[0], binary_data=True))
        onet_outputs.append(tritonclient.http.InferRequestedOutput(self.Onet_outputs[1], binary_data=True))
        onet_outputs.append(tritonclient.http.InferRequestedOutput(self.Onet_outputs[2], binary_data=True))

        output = self.triton_client.infer("onet", inputs=onet_inputs, outputs=onet_outputs)
        cls_prob, roi_prob, pts_prob = output.as_numpy(self.Onet_outputs[0]), output.as_numpy(self.Onet_outputs[1]), output.as_numpy(self.Onet_outputs[2])

        #-------------------------------------#
        #   解码的过程
        #-------------------------------------#
        # print('cls_prob:')
        # print(cls_prob)
        # print('roi_prob:')
        # print(roi_prob)
        # print('pts_prob:')
        # print(pts_prob)
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles


if __name__ == '__main__':
    model = mtcnn()
    #-----------------------------#
    #        设置检测门限
    #-----------------------------#
    threshold = [0.5, 0.6, 0.7]
    #-----------------------------#
    #        读取图片
    #-----------------------------#
    img = cv2.imread('test.jpg')
    temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #-----------------------------#
    #        将图片传入并检测
    #-----------------------------#
    t1 = time.time()
    rectangles = model.detectFace(temp_img, threshold)
    draw = img.copy()

    for rectangle in rectangles:
        W = int(rectangle[2]) - int(rectangle[0])
        H = int(rectangle[3]) - int(rectangle[1])

        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])),
                      (int(rectangle[2]), int(rectangle[3])), (0, 0, 255), 2)

        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i+0]), int(rectangle[i+1])), 1, (255, 0, 0), 4)
    
    t2 = time.time()
    print('inference time is: {}ms'.format(1000*(t2-t1)))
    cv2.imwrite('out.jpg', draw)
    # cv2.imshow('test', draw)
    # c = cv2.waitKey(0)


