# 追踪模型
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from utils.toolbox import Interpolator, Timer
from utils.regional_judgment import point_in_rect
from utils.draw import draw_box


class Track:
    """
    针对每帧结果的自定义追踪模型
    """

    def __init__(
            self,
            weight,
            imgsz=(640, 640),
            classes=[0, 2],
            tracker="bytetrack.yaml",
            verbose=False,
            show_fps=False,
            vid_stride=10,
            index=0
    ):
        # init params
        self.imgsz = imgsz
        self.classes = classes
        self.tracker = tracker
        self.verbose = verbose
        self.show_fps = show_fps

        self.model = YOLO(weight, task='detect')  # 35ms gpu

        # 300帧清空离场id
        self.timer = Timer(30)

        # 跳帧计算
        self.interpolator = Interpolator(vid_stride=vid_stride)

        self.vid_stride = vid_stride
        self.count = 0
        self.prior_result = None
        self.index = index
        self.detect = False

    def __call__(self, frame, show_id: dict, l_rate=None, r_rate=None):
        # click point
        w, h = frame.shape[1], frame.shape[0]
        l_point = (int(w * l_rate[0]), int(h * l_rate[1])) if l_rate is not None else None
        r_point = (int(w * r_rate[0]), int(h * r_rate[1])) if r_rate is not None else None
        # print('show_id==', show_id, l_point, r_point)

        if self.index == 0:
            if self.count == 0:
                self.detect = True
            else:
                self.detect = False
            self.count = (self.count + 1) % self.vid_stride
        elif self.index > 0:
            self.detect = False
            self.index -= 1
        # print('count', self.count, self.index)
            
        if self.detect:
            # inference
            results = self.model.track(
                    frame,
                    classes=self.classes,
                    tracker=self.tracker,
                    imgsz=self.imgsz,
                    half=True,
                    verbose=self.verbose,
                    persist=True
                )
            self.prior_result = results
        else:
            results = self.prior_result

        if results is None:
            return frame, show_id

        # maintain show_id
        try:
            id_set = set(results[0].boxes.id.int().cpu().tolist())
        except AttributeError:
            id_set = set()
        show_id = self.timer(id_set, show_id)

        # 中间帧插值
        det = results[0].boxes.data.cpu().numpy()
        # det = self.interpolator(det)

        # 自定义绘制
        # annotated_frame = draw_box(frame, det, results[0].names,
        #                            example=str(results[0].names))
        annotated_frame = frame.copy()
        annotator = Annotator(annotated_frame, line_width=4, example=str(results[0].names))

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
                    label = f"{id} {results[0].names[c]} {conf:.2f}"
                    # print('xyxy', det, xyxy)
                    annotator.box_label(xyxy, label, color=colors(c, True))

        return annotated_frame, show_id


if __name__ == '__main__':
    # test
    yolo = YOLO('yolov5s.pt')
    tracker = Track()
    for result in yolo.stream(r'F:\Projects\python\yolov8_stream\test_person30.mp4'):
        pass
