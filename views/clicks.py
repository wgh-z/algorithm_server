#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 鼠标操作相关路由
from flask import Blueprint, request, jsonify

mouse_operation = Blueprint('mouse_operation', __name__)

# 单击左键
@mouse_operation.route('/left', methods=['POST'])  # 参数要与html相同
def leftpointer():
    global l_rate
    x = float(request.args["xrate"])  # 接收客户端传来的参数
    y = float(request.args["yrate"])
    l_rate = (x, y)
    print('left==', l_rate)
    return "success"

# 双击左键
@mouse_operation.route('/double', methods=['POST'])  # 参数要与html相同
def doubleleftpointer():
    global r_rate
    x = float(request.args["xrate"])  # 接收客户端传来的参数
    y = float(request.args["yrate"])
    r_rate = (x, y)
    print('right==', r_rate)
    return "success"

@mouse_operation.route('/click', methods=['POST'])
def click():
    global l_rate
    x = float(request.json['x'])
    y = float(request.json['y'])
    l_rate = (x, y)
    return jsonify({'code':0})

@mouse_operation.route('/dblclick', methods=['POST'])
def dbclick():
    global r_rate
    x = float(request.json['x'])
    y = float(request.json['y'])
    r_rate = (x, y)
    return jsonify({'code':0})