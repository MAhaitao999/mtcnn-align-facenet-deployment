import time
from builtins import *

import cv2
import numpy as np

import tritonclient.http
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

import utils.utils as utils


class facenet(object):
    def __init__(self):
        self.triton_client = tritonclient.http.InferenceServerClient("127.0.0.1:8000", verbose=False)
        self.Facenet_inputs =  ['input_1']
        self.Facenet_outputs =  ['Bottleneck_BatchNorm']

    def calc_128_vec(self, img):
        face_img = utils.pre_process(img)
        # print('face image is: ', face_img.shape)
        facenet_inputs = []
        facenet_outputs = []
        facenet_inputs.append(tritonclient.http.InferInput(self.Facenet_inputs[0], face_img.shape, 'FP32'))
        facenet_inputs[0].set_data_from_numpy(face_img.astype(np.float32), binary_data=True)

        facenet_outputs.append(tritonclient.http.InferRequestedOutput(self.Facenet_outputs[0], binary_data=True))
        output = self.triton_client.infer("facenet_onnx", inputs=facenet_inputs, outputs=facenet_outputs)
        pre = output.as_numpy(self.Facenet_outputs[0])
        pre = utils.l2_normalize(np.concatenate(pre))
        pre = np.reshape(pre, [128])
        # print('pre is: ', pre.shape)
        
        return pre


if __name__ == '__main__':
    
    facenet_model = facenet()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, draw = video_capture.read()
        draw = np.expand_dims(cv2.resize(draw, (160, 160)), 0)
        feature = facenet_model.calc_128_vec(draw)
        cv2.imshow('Video', draw[0])
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
