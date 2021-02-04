#########################################################################
# File Name: detect.py
# Author: Henry Ma
# mail: iloveicRazavi@gmail.com
# Create Time: 2021年02月04日 星期四 15时55分20秒
#########################################################################

# !/usr/bin/python
# -*- coding:utf-8 -*-

from builtins import *

import cv2
import numpy as np

from mtcnn import mtcnn


if __name__ == '__main__':
    model = mtcnn()
    #---------------------------------------#
    #   设置检测门限
    #---------------------------------------#
    threshold = [0.5, 0.6, 0.7]
    #---------------------------------------#
    #   读取图片
    #---------------------------------------#
    img = cv2.imread('img/test.jpg')
    temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #-------------------------------------#
    #   将图片传入并检测
    #-------------------------------------#
    rectangles = model.detectFace(temp_img, threshold)

    draw = img.copy()
    for rectangle in rectangles:
        W = int(rectangle[2]) - int(rectangle[0])
        H = int(rectangle[3]) - int(rectangle[1])

        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])),
                      (int(rectangle[2]), int(rectangle[3])), (0, 0, 255), 2)

        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 1, (255, 0, 0), 4)

    cv2.imwrite("img/out.jpg", draw)
    cv2.imshow("test", draw)
    c = cv2.waitKey(0)
