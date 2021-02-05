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

TensorRT镜像选择的是`nvcr.io/nvidia/tensorrt:20.12-py3`（硬件环境和版本需要与Triton Server保持一致）

- pnet

```sh
trtexec --explicitBatch --workspace=512 --onnx=pnet.onnx \
--minShapes=input_1:1x12x12x3 \
--optShapes=input_1:4x400x400x3 \
--maxShapes=input_1:8x1280x720x3 \
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

把pnet.engine，rnet.engine，onet.engine 分别拷到repo对应的目录下，重命名成`model.plan`。

```sh
docker run --runtime=nvidia --network=host -it --name mtcnn-server -v `pwd`/repo:/repo nvcr.io/nvidia/tritonserver:20.12-py3 bash
```

在容器中执行`/opt/tritonserver/bin/tritonserver --model-store=/repo/ --log-verbose 1`命令。

打印如下日志说明部署成功：

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

### Triton Server客户端

起一个Triton Client容器：

```sh
docker run --runtime=nvidia --network=host -it --name mtcnn-client -v `pwd`/mtcnn_workspace:/mtcnn_workspace nvcr.io/nvidia/tritonserver:20.12-py3-sdk bash
```

用自带的`perf_client`工具测试一下server是否能正常工作：

```sh
./perf_client -m pnet --shape input_1:480,480,3
./perf_client -m rnet --shape input_1:24,24,3
./perf_client -m onet --shape input_1:48,48,3
```
