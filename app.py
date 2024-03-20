#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server的主程序，用于启动flask服务
import cv2
import yaml
import time
# from pathlib import Path
from flask import Flask, Response, render_template  # , request, jsonify
from flask_cors import CORS
from views.interactive import interactive_blueprint
from views.model_management import model_blueprint
from utils.backend import SmartBackend
from utils.video_io import generate


predictor = SmartBackend()

app = Flask(__name__)
app.register_blueprint(interactive_blueprint(predictor))
app.register_blueprint(model_blueprint(predictor))
CORS(app, supports_credentials=True)  # 允许跨域请求


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate(predictor),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask_app():
    global app
    with open('./cfg/server.yaml', 'r', encoding='utf-8') as f:
        server_cfg = yaml.load(f, Loader=yaml.FullLoader)
    app.run(host=server_cfg['ip'], port=server_cfg['port'], debug=False)


if __name__ == '__main__':
    run_flask_app()
