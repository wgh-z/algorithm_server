#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server的主程序，用于启动flask服务
import yaml
# from pathlib import Path
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2


app = Flask(__name__)
CORS(app, supports_credentials=True)

with open('cfg/frontend.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# 单击左键
@app.route('/left')  # 参数要与html相同
def leftpointer():
    global l_rate
    x = float(request.args["xrate"])  # 接收客户端传来的参数
    y = float(request.args["yrate"])
    l_rate = (x, y)
    print('left==', l_rate)
    return "success"

# 双击左键
@app.route('/double')  # 参数要与html相同
def doubleleftpointer():
    global r_rate
    x = float(request.args["xrate"])  # 接收客户端传来的参数
    y = float(request.args["yrate"])
    r_rate = (x, y)
    print('right==', r_rate)
    return "success"

@app.route('/click', methods=['POST'])
def click():
    global l_rate
    x = float(request.json['x'])
    y = float(request.json['y'])
    l_rate = (x, y)
    return jsonify({'code':0})

@app.route('/dblclick', methods=['POST'])
def dbclick():
    global r_rate
    x = float(request.json['x'])
    y = float(request.json['y'])
    r_rate = (x, y)
    return jsonify({'code':0})

@app.route('/api', methods=['POST'])
def api():
    data = request.json
    print(data)
    return jsonify({'code': 200, 'msg': 'success'})

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


if __name__ == '__main__':
    app.run(host=cfg.ip, port=cfg.port, debug=False)
