import os
import time

import cv2
import numpy as np

import utils.utils as utils
from mtcnn import mtcnn
from facenet import facenet  # tensorrt
# from facenet_onnx import facenet  # onnx
# from facenet_tf import facenet  # tensorflow


class face_rec(object):

    def __init__(self):
        #-------------------------#
        #   创建mtcnn的模型
        #   用于检测人脸
        #-------------------------#
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5, 0.6, 0.8]

        #-----------------------------------#
        #   载入facenet
        #   将检测到的人脸转化为128维的向量
        #-----------------------------------#
        self.facenet_model = facenet()

        #-----------------------------------------------#
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        #-----------------------------------------------#
        face_list = os.listdir('face_dataset')
        # print(face_list)
        self.known_face_encodings = []
        self.known_face_names = []
        for face in face_list:
            name = face.split('.')[0]
            img = cv2.imread('./face_dataset/' + face)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #---------------------#
            #   检测人脸
            #---------------------#
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)
            #---------------------#
            #   转化成正方形
            #---------------------#
            rectangles = utils.rect2square(np.array(rectangles))
            #-----------------------------------------------#
            #   facenet要传入一个160x160的图片
            #   利用landmark对人脸进行矫正
            #-----------------------------------------------#
            rectangle = rectangles[0]
            landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            # cv2.imwrite(name+'.jpg', crop_img)
            # print(crop_img.shape)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)
            #--------------------------------------------------------------------#
            #   将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            #--------------------------------------------------------------------#
            face_encoding = self.facenet_model.calc_128_vec(crop_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)

        # print(self.known_face_encodings)
        # print(self.known_face_names)

    def recognize(self, draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        height, width, _ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        #--------------------------------#
        #   检测人脸
        #--------------------------------#
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles)==0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)
        rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)

        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            #---------------#
            #   截取图像
            #---------------#
            landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #-----------------------------------------------#
            #   利用人脸关键点进行人脸对齐
            #-----------------------------------------------#
            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)

            face_encoding = self.facenet_model.calc_128_vec(crop_img)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            #-------------------------------------------------------#
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            #-------------------------------------------------------#
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.5)
            name = "Unknown"
            #-------------------------------------------------------#
            #   找出距离最近的人脸
            #-------------------------------------------------------#
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            #-------------------------------------------------------#
            #   取出这个最近人脸的评分
            #-------------------------------------------------------#
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        rectangles = rectangles[:, 0:4]
        #-----------------------------------------------#
        #   画框~!~
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2)
        return draw


if __name__ == "__main__":
    
    face = face_rec()
    video_capture = cv2.VideoCapture(0)

    while True:

        t1 = time.time()

        # 从摄像头读取图片
        success, img = video_capture.read()
        if not success:
            break

        draw = cv2.flip(img, 1)
        face.recognize(draw)
        
        t2 = time.time()
        
        fps = int(1. / (t2 - t1))
        # print('fps is: ', fps)
        position = (int(0.25 * draw.shape[0]), int(0.25 * draw.shape[1]))
        cv2.putText(draw, "FPS:{}".format(fps), position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
