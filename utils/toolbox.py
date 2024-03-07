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
    '''
    def __init__(self, vid_stride):
        self.vid_stride = vid_stride
        self.stride_counter = vid_stride
        self.prior_det = None

    def __call__(self, current_det):
        if self.stride_counter == self.vid_stride:
            # self.prior_det = current_det[:, :4]
            self.prior_det = current_det
            self.stride_counter = 0
        else:
            self.stride_counter += 1
            # current_det[:, :4] = interpolate_bbox(self.prior_det, current_det[:, :4], self.stride_counter)
            current_det = self.prior_det
        return current_det


class ClickFilterDet:
    '''
    这是一个点击过滤器类，用于使用点击过滤yolo检测结果
    '''
    def __init__(self, frame):
        self.frame = frame
        self.click_point = None

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
