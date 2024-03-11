#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server的主程序，用于启动flask服务
import cv2
import yaml
# from pathlib import Path
from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS
from views.clicks import create_blueprint # mouse_operation
from views.model_management import model_management
from utils.pridect import Pridect
import time


weight = r'E:\Projects\weight\yolo\v8\detect\coco\yolov8m.pt'
stream = 'list.streams'
imgsz = 640
predicter = Pridect(weight, stream, imgsz)

app = Flask(__name__)
app.register_blueprint(create_blueprint(predicter))
# app.register_blueprint(model_management)
CORS(app, supports_credentials=True)  # 允许跨域请求

with open('cfg/server.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    predicter.run()
    d_fps = 0
    while True:
        t1 = time.time()
        frame = predicter.collect_results()
        d_fps = (d_fps + (1 / (time.time() - t1))) / 2
        frame = cv2.putText(frame, f"FPS={d_fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 显示fps
        frame = cv2.resize(frame, (1280, 720))
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
