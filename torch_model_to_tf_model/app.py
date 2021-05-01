# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 20:22
# @Author  : zxf
import os
import json
import requests

import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


@app.route('/mlp', methods=['POST'])
def mlp_model_infer():
    params = json.loads(request.get_data(), encoding="utf-8")
    text = params.get("text")
    input_ = np.random.randn(1, 20).astype(np.float32).tolist()

    tensor = {"instances": [{"test_input": input_}]}
    tf_url = "http://192.168.4.31:15876/v1/models/mlp:predict"
    result = requests.post(tf_url, json=tensor)
    print("result: ", result)
    if result.status_code == 200:
        pred = result.json()['predictions'][0]
        reture_result = {"code": 200,
                         "message": "finish",
                         "result": pred}
        return jsonify(reture_result)
    else:
        reture_result = {"code": 200,
                         "message": "faile"}
        return jsonify(reture_result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
