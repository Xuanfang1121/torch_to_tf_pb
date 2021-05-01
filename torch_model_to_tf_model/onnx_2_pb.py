# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 20:15
# @Author  : zxf
import onnx
from onnx_tf.backend import prepare


model_onnx = onnx.load('./models/model_simple.onnx')

tf_rep = prepare(model_onnx)

# Export model as .pb file
tf_rep.export_graph('./models/model_simple.pb')
