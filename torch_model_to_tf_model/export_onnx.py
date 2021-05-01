# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 20:11
# @Author  : zxf
import torch

from model import SimpleModel


input_size = 20
hidden_sizes = [50, 50]
output_size = 1
num_classes = 2

model_pytorch = SimpleModel(input_size=input_size,
                            hidden_sizes=hidden_sizes,
                            output_size=output_size)
model_pytorch.load_state_dict(torch.load('./models/model_simple.pt'))

# dummy_input = torch.from_numpy(X_test[0].reshape(1, -1)).float().to(device)
dummy_input = torch.randn([1, 20])
dummy_output = model_pytorch(dummy_input)
print(dummy_output)

# Export to ONNX format
torch.onnx.export(model_pytorch, dummy_input, './models/model_simple.onnx',
                  input_names=['test_input'],
                  output_names=['test_output'])