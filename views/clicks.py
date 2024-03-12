#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 鼠标操作相关路由
from flask import Blueprint, request, jsonify


def clicks_blueprint(predicter):
    mouse_operation = Blueprint('mouse_operation', __name__)

    # 单击左键
    @mouse_operation.route('/clickleft', methods=['POST'])  # 参数要与html相同
    def leftpointer():
        global l_rate
        x = float(request.args["xrate"])  # 接收客户端传来的参数
        y = float(request.args["yrate"])
        l_rate = (x, y)
        print('left==', l_rate)
        return jsonify({'code': 200, 'msg': 'success'})

    # 双击左键
    @mouse_operation.route('/clickdouble', methods=['POST'])
    def doubleleftpointer():
        global r_rate
        x = float(request.args["xrate"])
        y = float(request.args["yrate"])
        r_rate = (x, y)
        print('right==', r_rate)
        return jsonify({'code': 200, 'msg': 'success'})

    # 单击左键
    @mouse_operation.route('/click', methods=['POST'])
    def click():
        global l_rate
        x = float(request.json['x'])
        y = float(request.json['y'])
        l_rate = (x, y)
        return jsonify({'code': 200, 'msg': 'success'})

    # 双击左键
    @mouse_operation.route('/dblclick', methods=['POST'])
    def dbclick():
        global r_rate
        x = float(request.json['x'])
        y = float(request.json['y'])
        r_rate = (x, y)
        return jsonify({'code': 200, 'msg': 'success'})

    # 左方向键
    @mouse_operation.route('/keyleft', methods=['POST'])
    def keyleft():
        print('keyleft')
        result = predicter.prior_group()
        print('result:', result)
        return jsonify({'code': 200, 'msg': 'success'})

    # 右方向键
    @mouse_operation.route('/keyright', methods=['POST'])
    def keyright():
        print('keyright')
        result = predicter.next_group()
        print('result:', result)
        return jsonify({'code': 200, 'msg': 'success'})
    
    return mouse_operation
