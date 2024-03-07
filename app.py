#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server的主程序，用于启动flask服务
from pathlib import Path
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2


app = Flask(__name__)
CORS(app, supports_credentials=True)

global cfg
cfg = Path('server/server_config.yaml').read_text().rsplit()

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
