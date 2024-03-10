# 工具箱
import numpy as np
from utils.regional_judgment import point_in_rect


def interpolate_bbox(bbox1, bbox2, n=1):
    # bbox转np.array
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    # 计算插值后的 bbox 坐标
    bbox_n = bbox2 + (bbox2 - bbox1)*n

    # 返回插值后的 bbox
    return bbox_n


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
