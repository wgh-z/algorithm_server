# 视频处理相关
import cv2
import time
import numpy as np
from utils.toolbox import create_void_img


class VideoDisplayManage:
    """
    这是一个视频显示管理类，用于管理视频显示
    """
    def __init__(self, group_scale, groups_num, scale) -> None:
        self.group_scale = group_scale
        self.groups_num = groups_num
        self.scale = scale

        self.intergroup_index = 0  # 组间索引
        self.intragroup_index = -1  # 组内索引。-1表示宫格显示，0-3表示单路显示
    
    def switch_group(self, direction: int = 1):
        assert direction in [-1, 1], 'direction must be -1 or 1'
        if self.intragroup_index == -1:  # 不响应 组内显示
            self.intergroup_index = (self.intergroup_index + direction) % self.groups_num
        else:
            self.intragroup_index = (self.intragroup_index + direction) % self.group_scale
        return f'当前显示第{self.intergroup_index}组视频'
    
    # 选择组内视频
    def select_intragroup(self, d_click_rate: tuple):
        x, y = d_click_rate
        self.intragroup_index = int(x//(1/self.scale) + (y//(1/self.scale))*self.scale)
        return f'当前显示第{self.intergroup_index}组，第{self.intragroup_index}路视频'

    def exit_intragroup(self):
        self.intragroup_index = -1
        return f'当前显示第{self.intergroup_index}组视频'
    
    def get_display_index(self):
        if self.intragroup_index == -1:
            return self.intergroup_index+1
        else:
            return self.intergroup_index+1, self.intragroup_index+1
    
    def reset(self):
        self.intergroup_index = 0
        self.exit_intragroup()


class ReadVideo:
    """
    这是一个视频读取类，用于读取视频
    """
    def __init__(self, source, timeout: int = 2) -> None:
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            w, h = int(self.cap.get(3)), int(self.cap.get(4))
        else:
            print(f'视频{self.source}打开失败')
            w, h = 1920, 1080
        self.void_img = create_void_img((w, h), '无信号')
        self.timeout = timeout
    
    # def __call__(self):
    #     success, frame = self.cap.read()
    #     if not success:
    #         start_time = time.time()  # 记录开始时间
    #         while not success:
    #             self.cap.release()
    #             self.cap = cv2.VideoCapture(self.source)
    #             success, frame = self.cap.read()
    #             if time.time() - start_time > self.timeout:  # 2s超时
    #                 print(f'视频{self.source}读取超时')
    #                 self.release()
    #                 return None
    #     return frame
    
    def __call__(self):
        success, frame = self.cap.read()
        if not success:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.source)
            return self.void_img, success
        return frame, success

    def release(self):
        self.cap.release()
        print(f'视频{self.source}已释放')
