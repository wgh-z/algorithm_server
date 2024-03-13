#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server的主程序，用于启动flask服务
import cv2
import yaml
import time
# from pathlib import Path
from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
from views.interactive import interactive_blueprint
from views.model_management import model_blueprint
from utils.backend import SmartBackend


with open('cfg/server.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

predicter = SmartBackend(
    cfg['weight'], cfg['sources'], cfg['imgsz'],
    group_scale=cfg['group_scale'], vid_stride=cfg['video_stride']
    )

app = Flask(__name__)
app.register_blueprint(interactive_blueprint(predicter))
app.register_blueprint(model_blueprint(predicter))
CORS(app, supports_credentials=True)  # 允许跨域请求

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    predicter.start()
    show_fps = 3
    sleep_time = 1 / show_fps
    while predicter.running:
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
