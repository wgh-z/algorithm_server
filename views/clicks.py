#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 鼠标操作相关路由
from flask import Blueprint, request, jsonify


def create_blueprint(predicter):
    mouse_operation = Blueprint('mouse_operation', __name__)

    # 单击左键
    @mouse_operation.route('/clickleft', methods=['get'])  # 参数要与html相同
    def leftpointer():
        global l_rate
        x = float(request.args["xrate"])  # 接收客户端传来的参数
        y = float(request.args["yrate"])
        l_rate = (x, y)
        print('left==', l_rate)
        return "success"

    # 双击左键
    @mouse_operation.route('/clickdouble', methods=['get'])
    def doubleleftpointer():
        global r_rate
        x = float(request.args["xrate"])
        y = float(request.args["yrate"])
        r_rate = (x, y)
        print('right==', r_rate)
        return "success"

    # 单击左键
    @mouse_operation.route('/click', methods=['POST'])
    def click():
        global l_rate
        x = float(request.json['x'])
        y = float(request.json['y'])
        l_rate = (x, y)
        return jsonify({'code':0})

    # 双击左键
    @mouse_operation.route('/dblclick', methods=['POST'])
    def dbclick():
        global r_rate
        x = float(request.json['x'])
        y = float(request.json['y'])
        r_rate = (x, y)
        return jsonify({'code':0})

    # 左方向键
    @mouse_operation.route('/keyleft', methods=['get'])
    def keyleft():
        print('keyleft')
        result = predicter.prior_group()
        print('result:', result)
        return jsonify({'code':0})

    # 右方向键
    @mouse_operation.route('/keyright', methods=['get'])
    def keyright():
        print('keyright')
        result = predicter.next_group()
        print('result:', result)
        return jsonify({'code':0})
    
    return mouse_operation
