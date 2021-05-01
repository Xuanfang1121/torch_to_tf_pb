# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 21:31
# @Author  : zxf
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants


def generator_tf_serving_pb_v1(export_dir, graph_pb):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


    sigs = {}
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name="")

        g = tf.get_default_graph()

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={"test_input": g.get_tensor_by_name('test_input:0')},
                outputs={"output": g.get_tensor_by_name('test_output:0')}
            )

        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],
                                             signature_def_map=sigs)

        builder.save()


if __name__ == "__main__":
    export_dir = './pb_model/1'
    graph_pb = './models/model_simple.pb'
    generator_tf_serving_pb_v1(export_dir, graph_pb)