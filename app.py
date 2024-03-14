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


predictor = SmartBackend()

app = Flask(__name__)
app.register_blueprint(interactive_blueprint(predictor))
app.register_blueprint(model_blueprint(predictor))
CORS(app, supports_credentials=True)  # 允许跨域请求


@app.route('/')
def index():
    return render_template('index.html')


def generate():
    predictor.start()
    show_fps = 25
    sleep_time = 1 / show_fps
    while predictor.running:
        t1 = time.time()
        frame = predictor.get_results()
        frame = cv2.resize(frame, (1280, 720))
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + \
              frame + b'\r\n'  # concat frame one by one and show result
        spend_time = time.time() - t1
        time.sleep(sleep_time-spend_time if spend_time < sleep_time else 0)

    print('帧获取结束')
    # return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + \
    #           b'' + b'\r\n'


@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask_app():
    global app
    with open('./cfg/server.yaml', 'r', encoding='utf-8') as f:
        server_cfg = yaml.load(f, Loader=yaml.FullLoader)
    app.run(host=server_cfg['ip'], port=server_cfg['port'], debug=False)


if __name__ == '__main__':
    run_flask_app()
