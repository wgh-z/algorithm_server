# 远程鼠标点击显示追踪目标
# import pyautogui
# import io
import yaml
import requests
import cv2 as cv
import numpy as np
# from PIL import Image
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from models.track import Track
from ultralytics import YOLO



app = Flask(__name__)
CORS(app, supports_credentials=True)

with open('cfg/frontend.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# 获取屏幕尺寸
# screenWidth, screenHeight = pyautogui.size()

l_rate, r_rate = None, None
show_id = dict()

@app.route('/')
def index():
    return render_template('index.html')

# 单击左键
@app.route('/left')  # 参数要与html相同
def leftpointer():
    global l_rate
    x = float(request.args["xrate"])  # 接收客户端传来的参数
    y = float(request.args["yrate"])
    l_rate = (x, y)
    print('left==', l_rate)
    return "success"

# # 单击右键
# @app.route('/right')
# def rightpointer():
#     global r_rate
#     x = float(request.args["xrate"])  # 接收客户端传来的参数
#     y = float(request.args["yrate"])
#     r_rate = (x, y)
#     print('right==', r_rate)
#     return "success"

# 双击左键
@app.route('/double')  # 参数要与html相同
def doubleleftpointer():
    global r_rate
    x = float(request.args["xrate"])  # 接收客户端传来的参数
    y = float(request.args["yrate"])
    r_rate = (x, y)
    print('right==', r_rate)
    return "success"

# # 按下
# @app.route('/down')
# def down():
#     global l_rate
#     x = float(request.args["xrate"])  # 接收客户端传来的参数
#     y = float(request.args["yrate"])
#     # pyautogui.mouseDown(x, y)   # 鼠标按下
#     l_rate = (x, y)
#     print('left==', x, y)
#     return "success"

# # 拖动
# @app.route('/move')
# def move():
#     x = int(float(request.args["xrate"]) * vidie_w)  # 接收客户端传来的参数
#     y = int(float(request.args["yrate"]) * video_h)
#     # pyautogui.moveTo(x, y)  # 拖动响应
#     print('move==', x, y)
#     return "success"

# # 释放
# @app.route('/up')
# def up():
#     global x, y
#     x = float(request.args["xrate"])  # 接收客户端传来的参数
#     y = float(request.args["yrate"])
#     # pyautogui.mouseUp()    # 鼠标释放
#     print('up==', x, y)
#     return "success"

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

def gen():
    global l_rate, r_rate, show_id, results, vid_stride

    l_count = 5
    r_count = 5

    tracker = Track(vid_stride=vid_stride)

    for frame in results:
    
        annotated_frame, show_id = tracker(frame, show_id, l_rate, r_rate)
        
        # point delay 10 frames
        if l_rate is not None:
            if l_count > 0:
                l_count -= 1
                cv.circle(annotated_frame, (int(frame.shape[1] * l_rate[0]), int(frame.shape[0] * l_rate[1])), 10, (0, 255, 0), -1)
            else:
                l_rate = None
                l_count = 5
        if r_rate is not None:
            if r_count > 0:
                r_count -= 1
                cv.circle(annotated_frame, (int(frame.shape[1] * r_rate[0]), int(frame.shape[0] * r_rate[1])), 10, (0, 0, 255), -1)
            else:
                r_rate = None
                r_count = 5

        ret, buffer = cv.imencode('.jpg', annotated_frame)
        out_frame = buffer.tobytes()
        yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + out_frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    stream = r'E:\Projects\test_data\video\MOT17-test\MOT17-01.mp4'
    vid_stride = 3
    weight = r'E:\Projects\weight\yolo\v8\detect\coco\yolov8m.pt'


    assert cfg.ip, "请输入ip地址"
    assert cfg.port, "请输入端口号"
    assert weight, "请输入权重文件"
    assert stream, "请输入视频流地址"

    if vid_stride == '':
        vid_stride = 1

    model_end = weight.split('_')[-1]
    imgsz = [int(size) for size in model_end.split('x')] if 'x' in model_end else [640, 640]

    model = YOLO(weight, task='detect')

    results = model.track(
        source=stream,
        classes=[0, 2],
        tracker="bytetrack.yaml",  # 20fps
        imgsz=imgsz,
        stream=True,
        verbose=False
        )  # 生成器

    app.run(host=cfg.ip, port=cfg.port, debug=False)
