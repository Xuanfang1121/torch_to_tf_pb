# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 20:36
# @Author  : zxf
import torch
import numpy as np
import tensorflow as tf


def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


tf_graph = load_pb('./models/model_simple.pb')
sess = tf.Session(graph=tf_graph)

# Show tensor names in graph
for op in tf_graph.get_operations():
  print(op.values())

output_tensor = tf_graph.get_tensor_by_name('test_output:0')
input_tensor = tf_graph.get_tensor_by_name('test_input:0')

dummy_input = np.random.randn(1, 20).astype(np.float32)
# dummy_input = torch.from_numpy(X_test[0].reshape(1, -1)).float().to(device)
output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
print(output)