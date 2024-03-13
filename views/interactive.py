#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 鼠标、键盘等交互式操作相关路由
from flask import Blueprint, request, jsonify


def interactive_blueprint(predicter):
    interactive_operation = Blueprint('mouse_operation', __name__)

    # 双击左键进入组内显示
    @interactive_operation.route('/clickdouble', methods=['POST'])
    def doubleleftpointer():
        x = float(request.args["xrate"])
        y = float(request.args["yrate"])
        d_click_rate = (x, y)
        print('doubleleft==', d_click_rate)
        result = predicter.select_intragroup(d_click_rate)
        return jsonify({'code': 200, 'msg': result})

    # 双击左键进入组内显示
    @interactive_operation.route('/dblclick', methods=['POST'])
    def dbclick():
        x = float(request.json['x'])
        y = float(request.json['y'])
        d_click_rate = (x, y)
        print('doubleleft==', d_click_rate)
        result = predicter.select_intragroup(d_click_rate)
        return jsonify({'code': 200, 'msg': result})
    
    # q键退出组内显示
    @interactive_operation.route('/keyq', methods=['POST'])
    def keyq():
        print('keyq')
        result = predicter.exit_intragroup()
        print(result)
        return jsonify({'code': 200, 'msg': result})

    # 左方向键循环切换上一组
    @interactive_operation.route('/keyleft', methods=['POST'])
    def keyleft():
        print('keyleft')
        result = predicter.prior_group()
        print(result)
        return jsonify({'code': 200, 'msg': result})

    # 右方向键循环切换下一组
    @interactive_operation.route('/keyright', methods=['POST'])
    def keyright():
        print('keyright')
        result = predicter.next_group()
        print(result)
        return jsonify({'code': 200, 'msg': result})
    
    # e键编辑检测区域
    @interactive_operation.route('/keye', methods=['POST'])
    def keye():
        print('keye')
        # predicter.select_in_group((0, 0))
        return jsonify({'code': 200, 'msg': 'success'})

    # 单击左键再按e添加检测区域点
    @interactive_operation.route('/clickleft', methods=['POST'])  # 参数要与html相同
    def leftpointer():
        x = float(request.args["xrate"])  # 接收客户端传来的参数
        y = float(request.args["yrate"])
        l_rate = (x, y)
        print('left==', l_rate)
        return jsonify({'code': 200, 'msg': 'success'})
    
    # 单击左键再按e添加检测区域点
    @interactive_operation.route('/click', methods=['POST'])
    def click():
        x = float(request.json['x'])
        y = float(request.json['y'])
        l_rate = (x, y)
        return jsonify({'code': 200, 'msg': 'success'})


    return interactive_operation
