# 视频处理相关
import cv2
import time


class VideoDisplayManage:
    '''
    这是一个视频显示管理类，用于管理视频显示
    '''
    def __init__(self, groups_num, scale) -> None:
        self.groups_num = groups_num
        self.scale = scale

        self.intergroup_index = 0  # 组间索引
        self.intragroup_index = -1  # 组内索引。-1表示宫格显示，0-3表示单路显示
    
    def switch_group(self, direction: int = 1):
        assert direction in [-1, 1], 'direction must be -1 or 1'
        if self.intragroup_index == -1:  # 不响应 组内显示
            self.intergroup_index = (self.intergroup_index + direction) % self.groups_num
        return f'当前显示第{self.intergroup_index}组视频'
    
    # 选择组内视频
    def select_intragroup(self, d_click_rate: tuple):
        x, y = d_click_rate
        self.intragroup_index = int(x//(1/self.scale) + (y//(1/self.scale))*self.scale)
        return f'当前显示第{self.intergroup_index}组，第{self.intragroup_index}路视频'

    def exit_intragroup(self):
        self.intragroup_index = -1
        return f'当前显示第{self.intergroup_index}组视频'
    
    def reset(self):
        self.intergroup_index = 0
        self.exit_intragroup()


class ReadVideo:
    '''
    这是一个视频读取类，用于读取视频
    '''
    def __init__(self, source, timeout: int = 2) -> None:
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.timeout = timeout
    
    def __call__(self):
        success, frame = self.cap.read()
        if not success:
            start_time = time.time()  # 记录开始时间
            while not success:
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source)
                success, frame = self.cap.read()
                if time.time() - start_time > self.timeout:  # 2s超时
                    print(f'视频{self.source}读取超时')
                    return None
        return frame
