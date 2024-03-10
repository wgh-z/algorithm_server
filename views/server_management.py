#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 算法服务器管理相关路由
from flask import Blueprint, request, jsonify

server_management = Blueprint('server_management', __name__)