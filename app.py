#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server的主程序，用于启动flask服务
import cv2
import yaml
# from pathlib import Path
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from views.clicks import mouse_operation
from views.model_management import model_management


app = Flask(__name__)
app.register_blueprint(mouse_operation)
app.register_blueprint(model_management)
CORS(app, supports_credentials=True)  # 允许跨域请求

with open('cfg/server.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def generate():
    cap = cv2.VideoCapture(0)  # 使用OpenCV从摄像头捕获视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask_app():
    global cfg, app
    app.run(host=cfg['ip'], port=cfg['port'], debug=False)

if __name__ == '__main__':
    run_flask_app()
