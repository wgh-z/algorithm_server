#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 模型管理相关路由
from flask import Blueprint, request, jsonify

model_management = Blueprint('model_management', __name__)

@model_management.route('/config', methods=['POST'])
# 配置模型参数
def api():
    data = request.json
    print(data)
    return jsonify({'code': 200, 'msg': 'success'})

@model_management.route('/reboot', methods=['POST'])
# 重启模型
def api():
    data = request.json
    print(data)
    return jsonify({'code': 200, 'msg': 'success'})

@model_management.route('/shutdown', methods=['POST'])
# 关闭模型
def api():
    data = request.json
    print(data)
    return jsonify({'code': 200, 'msg': 'success'})

@model_management.route('/update', methods=['POST'])
# 更新模型
def api():
    data = request.json
    print(data)
    return jsonify({'code': 200, 'msg': 'success'})

@model_management.route('/upload', methods=['POST'])
# 上传模型
def api():
    data = request.json
    print(data)
    return jsonify({'code': 200, 'msg': 'success'})

@model_management.route('/download', methods=['POST'])
# 下载模型
def api():
    data = request.json
    print(data)
    return jsonify({'code': 200, 'msg': 'success'})