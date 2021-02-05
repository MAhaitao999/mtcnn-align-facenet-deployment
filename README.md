# mtcnn-onnx

本项目参考了bubbliiiing的[mtcnn-keras](https://github.com/bubbliiiing/mtcnn-keras)工程。

它的是keras模型，我用keras2onnx工具把它转换成了onnx模型，其他胶水部分的逻辑没什么变化。

添加了一个对摄像头读取视频进行检测的支持，详情请参考`detect_video.py`文件.

推理速度的话我在我的DELL笔记本电脑上(纯CPU环境)测了一下，之前的keras模型FPS差不多为6，onnx模型FPS为8，提升了30%左右。

### keras模型转onnx模型

mtcnn-keras提供的模型文件只有权重，因此需要先结合网络结构把它变成结构和权重均有的模型。然后再用keras2onnx工具将其转换成onnx模型。

```sh
python3 keras_onnx.py
```

### 采用onnx模型进行推理

- 单张图片测试: `python3 detect.py`

- 调用摄像头: `python3 detect_video.py`

### onnx模型转trt模型

- pnet

```sh
trtexec --explicitBatch --workspace=512 --onnx=pnet.onnx \
--minShapes=input_1:1x100x100x3 \
--optShapes=input_1:4x400x400x3 \
--maxShapes=input_1:8x1080x720x3 \
--shapes=input_1:8x400x400x3 \
--saveEngine=pnet.engine
```

- rnet

```sh
trtexec --explicitBatch --workspace=512 --onnx=rnet.onnx \
--minShapes=input_1:1x24x24x3 \
--optShapes=input_1:64x24x24x3 \
--maxShapes=input_1:128x24x24x3 \
--shapes=input_1:64x24x24x3 \
--saveEngine=rnet.engine
```

- onet

```sh
trtexec --explicitBatch --workspace=512 --onnx=onet.onnx \
--minShapes=input_1:1x48x48x3 \
--optShapes=input_1:64x48x48x3 \
--maxShapes=input_1:128x48x48x3 \
--shapes=input_1:64x48x48x3 \
--saveEngine=onet.engine
```

### trt模型部署在Triton Server上

### Triton Server客户端


