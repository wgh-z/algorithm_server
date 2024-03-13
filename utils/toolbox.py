# 工具箱
import cv2
import time
import numpy as np
from utils.regional_judgment import point_in_rect

def fps(func):
    """
    这是一个计算帧率的装饰器
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('fps:', 1 / (time.time() - start))
        return result
    return wrapper

def display_avg_fps_decorator(func):
    def wrapper(*args, **kwargs):
        avg_fps = 0
        t1 = time.time()
        result = func(*args, **kwargs)
        self = args[0]  # 获取self参数
        avg_fps = (avg_fps + (self.vid_stride / (time.time() - t1))) / 2
        self.im_show = cv2.putText(self.im_show, f"FPS={avg_fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 显示fps
        return result
    return wrapper

def interpolate_bbox(bbox1, bbox2, n=1):
    # bbox转np.array
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    # 计算插值后的 bbox 坐标
    bbox_n = bbox2 + (bbox2 - bbox1)*n

    # 返回插值后的 bbox
    return bbox_n


class SquareSplice:
    """
    这是一个拼接图片类，将一组图片按平方宫格拼接成一张大图
    """
    def __init__(self, scale: int = 2, show_shape: tuple = (1920, 1080)):
        self.scale = scale
        self.show_w, self.show_h = show_shape
        self.grid_w = int(self.show_w / self.scale)
        self.grid_h = int(self.show_h / self.scale)

    def __call__(self, im_list):
        im = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        for i, im0 in enumerate(im_list):
            im0 = cv2.resize(im0, (self.grid_w, self.grid_h))
            im[self.grid_h*(i//self.scale):self.grid_h*(1+(i//self.scale)),
               self.grid_w*(i%self.scale):self.grid_w*(1+(i%self.scale))] = im0
        return im


class Timer:
    """
    这是一个计时器类，用于将(data_dict.keys() - data_dict.keys() ∩ data_set)的元素延迟delay_count次后删除
    """
    def __init__(self, delay_count:int = 30):
        self.delay_count = delay_count
    
    def add_delay(self, data_dict:dict, id:int):
        data_dict[id] = self.delay_count
        return data_dict

    def __call__(self, data_set:set, data_dict:dict):
        temp_dict = data_dict.copy()
        for element in data_dict.keys():
            if element not in data_set:
                temp_dict[element] -= 1
                if temp_dict[element] == 0:
                    del temp_dict[element]
            else:
                data_dict[element] = self.delay_count  # 重置计时器
        return temp_dict


class Interpolator:
    '''
    这是一个插值器类，用于对检测结果进行插值
    参数：
        vid_stride: int, 视频检测间隔
        mode: str, 插值模式： copy, linear, quadratic, cubic
    '''
    def __init__(self, vid_stride:int=2, mode:str='copy'):
        self.vid_stride = vid_stride
        self.stride_counter = vid_stride
        self.mode = mode
        self.prior_det = None

    def __call__(self, current_det):
        if self.vid_stride == 1:  # vid_stride为1时，不进行插值
            return current_det
        if self.mode == 'copy':
            return self.copy(current_det)
    
    def copy(self, current_det):
        '''
        超快速插值模式，即不插值，直接返回上一帧检测结果
        '''
        if self.stride_counter == self.vid_stride:
                # self.prior_det = current_det[:, :4]
                self.prior_det = current_det
                self.stride_counter = 0
        else:
            self.stride_counter += 1
            # current_det[:, :4] = interpolate_bbox(self.prior_det, current_det[:, :4], self.stride_counter)
            current_det = self.prior_det
        return current_det
    
    def balanced(self, current_det):
        '''
        平衡插值模式，即每隔vid_stride帧进行一次插值
        '''
        if self.stride_counter == self.vid_stride:
            if self.prior_det is not None:
                current_det = interpolate_bbox(self.prior_det, current_det, self.stride_counter)
            self.prior_det = current_det
            self.stride_counter = 0
        else:
            self.stride_counter += 1
        return current_det
    
    def slow(self, current_det):
        '''
        慢速插值模式，即每帧都进行插值
        '''
        if self.prior_det is not None:
            current_det = interpolate_bbox(self.prior_det, current_det, self.stride_counter)
        self.prior_det = current_det
        return current_det


class ClickFilterDet:
    '''
    这是一个点击过滤器类，用于使用点击坐标过滤或回恢复yolo检测结果
    '''
    def __init__(self, frame):
        self.frame = frame
        self.click_point = None

        # 30帧清空离场id
        self.timer = Timer(30)


    def __call__(self, det, l_point=None, r_point=None):
        for i, *xyxy in enumerate(reversed(det[:, :4])):
            if point_in_rect(l_point, xyxy):
                    # show_id.append(id)
                    show_id = self.timer.add_delay(show_id, id)
                    l_point = None

            if point_in_rect(r_point, xyxy):
                    try:
                        # show_id.remove(id)
                        del show_id[id]
                    except:
                        pass
                    r_point = None
            

        return l_point, r_point


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
    

class VideoRead:
    '''
    这是一个视频读取类，用于读取视频
    '''
    def __init__(self, source, stride: int = 1) -> None:
        self.source = source
        self.stride = stride
        self.cap = cv2.VideoCapture(self.source)
    
    def __call__(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.source)
            return self()
