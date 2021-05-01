## Torch模型转pb模型demo
1. requirements
```
torch==1.6.0
tensorflow==1.14.0
onnx==1.6.0
onnx-tf==1.5.0
```

2. 代码执行说明
```
(1) 先运行train.py
(2) 执行export_onnx.py
(3) 执行onnx_2_pb.py
(4) 执行get_tf_serving.py
(5) http请求代码
```