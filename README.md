# mtcnn-align-facenet-deployment

### 项目简介

本项目参考了bubbliiiing的[mtcnn-keras](https://github.com/bubbliiiing/mtcnn-keras)和[keras-face-recognition](https://github.com/bubbliiiing/keras-face-recognition)两个工程, 在此对作者表示感谢!

这两个工程都是keras模型, 所提供的模型文件都只有权重没有网络结构, 我利用作者提供的网络定义和权重文件重新生成了带有网络结构的权重文件. 比如原先只有权重的模型文件`pnet.h5`,
生成含网络结构和权重的模型文件`PNET.h5`. 接着用keras2onnx工具把它(`PNET.h5`)转换成了onnx模型`pnet.onnx`, 其他胶水部分的逻辑没什么变化. 具体的转换代码请参考`keras_onnx.py`文件.

另外我还尝试了将keras h5模型转成tensorflow pb模型, 具体代码请参考`h5_to_pb.py`文件. 需要注意的是: 每个tensorflow pb模型请**单独**执行`h5_to_pb.py`脚本生成. (每次修改weight_file参数)

如果你想简单地测试一下mtcnn人脸检测的效果, 请执行`python3 detect.py`. 如果你想调用你本地电脑的摄像头, 请执行`python3 detect_video.py`. 这两个都是onnxruntime调用onnx模型文件进行推理的.
需要注意的是: 你需要**先执行一下**`keras_onnx.py`来生成onnx模型文件.

自己的人脸底库请放置在`face_recognition_client/face_dataset/`目录下, 每张图片中只能有一张人脸, 该人的名字即为图片名.

本项目的重点是将模型转成TensorRT模型, 之后部署在Triton Server上, 然后编写客户端代码对几个模型的推理进行串联实现人脸识别的功能.

接下来将会按照以下几个方面做介绍:

- keras模型转onnx模型

- onnx模型转trt模型

- trt模型部署在Triton Server上

- Triton Client端调用实现人脸检测和识别功能

- onnx模型和tf模型部署

### keras模型转onnx模型

`pnet.h5`, `rnet.h5`和`onet.h5`三个文件比较小, 我已经放在了`model_data`目录下. `facenet.h5`文件比较大请自行前往百度云盘进行下载:

链接: https://pan.baidu.com/s/1A9jCJa_sQ4D3ejelgXX2RQ 提取码: tkhg

请将下载后的文件改名为`facenet.h5`, 放置于`model_data/`目录下.

执行脚本:

```py
python3 keras_onnx.py
```

会在`model_data/`目录下生成`pnet.onnx`, `rnet.onnx`, `onet.onnx`和`facenet.onnx`四个onnx模型文件. 同时还生成了`PNET.h5`, `RNET.h5`, `ONET.h5`和`FACENET.h5`四个带有网络结构和权重的keras模型文件.

### onnx模型转trt模型

我选用的TensorRT环境是`nvcr.io/nvidia/tensorrt:20.12-py3`, 需要注意的是: 你想在什么样的Triton Server环境中部署你的模型, 你就必须在对应的TensorRT环境中进行模型转换.(软硬件环境均需一致)

换句话说, 我的部署环境是`nvcr.io/nvidia/nvcr.io/nvidia/tritonserver:20.12-py3`, GPU 1050Ti, 则转换环境必须也是`nvcr.io/nvidia/tensorrt:20.12-py3`, GPU 1050Ti, 两个镜像的tag必须保持一致.

- pnet

```sh
trtexec --explicitBatch --workspace=512 --onnx=pnet.onnx \
--minShapes=input_1:1x12x12x3 \
--optShapes=input_1:4x400x400x3 \
--maxShapes=input_1:8x1280x1280x3 \
--shapes=input_1:8x400x400x3 \
--saveEngine=pnet.engine
```

- rnet

```sh
trtexec --explicitBatch --workspace=512 --onnx=rnet.onnx \
--minShapes=input_1:1x24x24x3 \
--optShapes=input_1:512x24x24x3 \
--maxShapes=input_1:4096x24x24x3 \
--shapes=input_1:512x24x24x3 \
--saveEngine=rnet.engine
```

- onet

```sh
trtexec --explicitBatch --workspace=512 --onnx=onet.onnx \
--minShapes=input_1:1x48x48x3 \
--optShapes=input_1:64x48x48x3 \
--maxShapes=input_1:256x48x48x3 \
--shapes=input_1:64x48x48x3 \
--saveEngine=onet.engine
```

- facenet

```sh
trtexec --explicitBatch --workspace=512 --onnx=facenet.onnx \
--minShapes=input_1:1x160x160x3 \
--optShapes=input_1:8x160x160x3 \
--maxShapes=input_1:16x160x160x3 \
--shapes=input_1:8x160x160x3 \
--saveEngine=facenet.engine
```

#### trt模型部署在Triton Server上

将`pnet.engine`, `rnet.engine`, `onet.engine`和`facenet.engine`分别拷贝到repo/对应的目录下, 重命名成`model.plan`. 对应的模型配置文件我已经都放在了`repo/`各个模型目录下.

```sh
docker run --runtime=nvidia --network=host -it --name mtcnn-server -v `pwd`/repo:/repo nvcr.io/nvidia/tritonserver:20.12-py3 bash
```

在容器中执行`/opt/tritonserver/bin/tritonserver --model-store=/repo/ --log-verbose 1`命令.

打印如下日志说明部署成功:

```
......
I0205 07:20:30.326955 159 grpc_server.cc:225] Ready for RPC 'ServerLive', 0
I0205 07:20:30.326992 159 grpc_server.cc:225] Ready for RPC 'ServerReady', 0
I0205 07:20:30.327006 159 grpc_server.cc:225] Ready for RPC 'ModelReady', 0
I0205 07:20:30.327018 159 grpc_server.cc:225] Ready for RPC 'ServerMetadata', 0
I0205 07:20:30.327041 159 grpc_server.cc:225] Ready for RPC 'ModelMetadata', 0
I0205 07:20:30.327055 159 grpc_server.cc:225] Ready for RPC 'ModelConfig', 0
I0205 07:20:30.327065 159 grpc_server.cc:225] Ready for RPC 'ModelStatistics', 0
I0205 07:20:30.327078 159 grpc_server.cc:225] Ready for RPC 'SystemSharedMemoryStatus', 0
I0205 07:20:30.327091 159 grpc_server.cc:225] Ready for RPC 'SystemSharedMemoryRegister', 0
I0205 07:20:30.327104 159 grpc_server.cc:225] Ready for RPC 'SystemSharedMemoryUnregister', 0
I0205 07:20:30.327116 159 grpc_server.cc:225] Ready for RPC 'CudaSharedMemoryStatus', 0
I0205 07:20:30.327128 159 grpc_server.cc:225] Ready for RPC 'CudaSharedMemoryRegister', 0
I0205 07:20:30.327141 159 grpc_server.cc:225] Ready for RPC 'CudaSharedMemoryUnregister', 0
I0205 07:20:30.327156 159 grpc_server.cc:225] Ready for RPC 'RepositoryIndex', 0
I0205 07:20:30.327166 159 grpc_server.cc:225] Ready for RPC 'RepositoryModelLoad', 0
I0205 07:20:30.327177 159 grpc_server.cc:225] Ready for RPC 'RepositoryModelUnload', 0
I0205 07:20:30.327200 159 grpc_server.cc:416] Thread started for CommonHandler
I0205 07:20:30.327347 159 grpc_server.cc:3082] New request handler for ModelInferHandler, 1
I0205 07:20:30.327368 159 grpc_server.cc:2146] Thread started for ModelInferHandler
I0205 07:20:30.327542 159 grpc_server.cc:3427] New request handler for ModelStreamInferHandler, 3
I0205 07:20:30.327586 159 grpc_server.cc:2146] Thread started for ModelStreamInferHandler
I0205 07:20:30.327595 159 grpc_server.cc:3979] Started GRPCInferenceService at 0.0.0.0:8001
I0205 07:20:30.327974 159 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I0205 07:20:30.369874 159 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```

### Triton Client端调用实现人脸检测和识别功能

起一个Triton Client容器:

```sh
docker run --runtime=nvidia --network=host \
--privileged -v /dev/video0:/dev/video0 -v /dev/video1:/dev/video1 \
-it --name mtcnn-client -v `pwd`:/mtcnn_workspace nvcr.io/nvidia/tritonserver:20.12-py3-sdk bash
```

为了在容器中调用主机的摄像头, 必须添加`--privileged -v /dev/video0:/dev/video0 -v /dev/video1:/dev/video1`几个选项.

用自带的`perf_client`工具测试一下模型服务是否能正常工作:

```sh
./perf_client -m pnet --shape input_1:480,480,3
./perf_client -m rnet --shape input_1:24,24,3
./perf_client -m onet --shape input_1:48,48,3
./perf_client -m facenet --shape input_1:160,160,3
```

用自己编写的客户端进行测试, 调用本机的摄像头进行人脸检测和人脸识别. 在容器中调用主机的摄像头还需要进行一些设置:

```sh
# 在主机上执行
xhost +

# 在容器中执行
export DISPLAY=:0.0
export QT_X11_NO_MITSHM=1
```

执行如下脚本, 调用本机摄像头进行人脸检测和人脸识别:

```sh
# 人脸检测: face_detection_client/目录下
python3 face_detection.py

# 人脸识别: face_recognition_client/目录下
python3 face_recognize.py
```

### onnx模型和tf模型部署

执行脚本:

```sh
python3 h5_to_pb.py
```

生成tf模型文件.

将model_data/目录下`pnet.onnx`, `rnet.onnx`, `onet.onnx`和`facenet.onnx`分别拷贝到repo/对应的目录下, 重命名成`model.onnx`. 
将model_data/目录下`pnet.pb`, `rnet.pb`, `onet.pb`和`facenet.pb`分别拷贝到repo/对应的目录下, 重命名成`model.grapgdef`.

注意: facenet_tf的配置文件中输出name不是`output_1`, 而是`Bottleneck_BatchNorm/cond/Merge`.

服务启动和客户端调用与上面类似, 不再赘述.
