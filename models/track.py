# 追踪模型
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from utils.toolbox import Interpolator, Timer
from utils.regional_judgment import point_in_rect


class Track:
    '''
    针对生成器结果的自定义追踪模型
    '''
    def __init__(
            self,
            show_fps=False,
            vid_stride=1
            ):
        # init params
        self.show_fps = show_fps

        # 300帧清空离场id
        self.timer = Timer(30)

        # 跳帧计算
        self.interpolator = Interpolator(vid_stride=vid_stride)

    def __call__(self, result, show_id:dict, l_rate=None, r_rate=None):
        # click point
        frame = result.orig_img
        h, w = result.orig_shape

        l_point = (int(w * l_rate[0]), int(h * l_rate[1])) if l_rate is not None else None
        r_point = (int(w * r_rate[0]), int(h * r_rate[1])) if r_rate is not None else None
        # print('show_id==', show_id, l_point, r_point)

        # maintain show_id
        try:
            id_set = set(result.boxes.id.int().cpu().tolist())
        except AttributeError:
            id_set = set()
        show_id = self.timer(id_set, show_id)

        # 自定义绘制
        annotated_frame = frame.copy()
        annotator = Annotator(annotated_frame, line_width=2, example=str(result.names))
        
        # 中间帧插值
        det = self.interpolator(result.boxes.data.cpu().numpy())
        if len(det) and len(det[0]) == 7:
            for *xyxy, id, conf, cls in reversed(det):
                c = int(cls)  # integer class
                id = int(id)  # integer id

                if l_point is not None and id not in show_id:
                    if point_in_rect(l_point, xyxy):
                        # show_id.append(id)
                        show_id = self.timer.add_delay(show_id, id)
                        l_point = None

                if r_point is not None:
                    if point_in_rect(r_point, xyxy):
                        try:
                            # show_id.remove(id)
                            del show_id[id]
                        except:
                            pass
                        r_point = None

                # 显示指定id的目标
                if id in show_id.keys() or show_id == {}:
                    label = f"{id} {result.names[c]} {conf:.2f}"
                    # print('xyxy', det, xyxy)
                    annotator.box_label(xyxy, label, color=colors(c, True))

        annotated_frame = annotator.result()
        return annotated_frame, show_id


if __name__ == '__main__':
    # test
    yolo = YOLO('yolov5s.pt')
    tracker = Track()
    for result in yolo.stream(r'F:\Projects\python\yolov8_stream\test_person30.mp4'):
        pass