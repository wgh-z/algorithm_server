#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server的主程序，用于启动flask服务
import cv2
import yaml
# from pathlib import Path
from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS
from views.clicks import clicks_blueprint # mouse_operation
from views.model_management import model_blueprint # model_management
from utils.pridect import Pridect
import time


weight = r'E:\Projects\weight\yolo\v8\detect\coco\yolov8m.pt'
stream = 'list.streams'
imgsz = 640
predicter = Pridect(weight, stream, imgsz, vid_stride=5)

app = Flask(__name__)
app.register_blueprint(clicks_blueprint(predicter))
app.register_blueprint(model_blueprint(predicter))
CORS(app, supports_credentials=True)  # 允许跨域请求

with open('cfg/server.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    predicter.start()
    show_fps = 3
    sleep_time = 1 / show_fps
    while predicter.run:
        frame = predicter.get_results()
        frame = cv2.resize(frame, (1280, 720))
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        time.sleep(sleep_time)
    print('帧获取结束')

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask_app():
    global cfg, app
    app.run(host=cfg['ip'], port=cfg['port'], debug=False)

if __name__ == '__main__':
    run_flask_app()
