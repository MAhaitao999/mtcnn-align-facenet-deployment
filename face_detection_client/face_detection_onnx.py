#########################################################################
# File Name: single_client.py
# Author: Henry Ma
# mail: iloveicRazavi@gmail.com
# Create Time: 2021年02月05日 星期五 16时01分13秒
#########################################################################

# !/usr/bin/python
# -*- coding:utf-8 -*-

import time

import cv2
import numpy as np

from mtcnn_onnx import mtcnn


if __name__ == "__main__":
    model = mtcnn()
    #-------------------------------------#
    #   设置检测门限
    #-------------------------------------#
    threshold = [0.5, 0.6, 0.7]
    #-------------------------------------#
    #   读取摄像头
    #-------------------------------------#
    cap = cv2.VideoCapture(0)
    while True:

        t1 = time.time()

        # 从摄像头读取图片
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
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
                          (int(rectangle[2]), int(rectangle[3])),
                          (0, 0, 255), 2)

            for i in range(5, 15, 2):
                cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 1, (255, 0, 0), 4)

        t2 = time.time()
        fps = int(1. / (t2 - t1))
        # print('fps is: ', fps)
        position = (int(0.25 * draw.shape[0]), int(0.25 * draw.shape[1]))
        cv2.putText(draw, "FPS:{}".format(fps), position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        cv2.imshow("test", draw)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k==ord("s"):
            # 通过s键保存图片，并退出
            cv2.imwrite('face.jpg', img)
            cv2.destroyAllWindows()
            break
    # 关闭摄像头
    cap.release()

