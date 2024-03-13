# 自定义模型预测相关
import cv2
import time
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.loaders import LoadStreams
from ultralytics.data.augment import LetterBox
from queue import Queue

from utils.toolbox import SquareSplice
from models.track import Track

class SmartBackend:
    def __init__(
            self,
            weight,
            source,
            imgsz,
            group_scale: int = 4,  # 每组4路视频
            show_w: int = 1920,
            show_h: int = 1080,
            vid_stride: int = 1
            ) -> None:
        self.weight = weight
        self.source = source
        self.imgsz = imgsz
        self.group_scale = group_scale
        self.show_w = show_w
        self.show_h = show_h
        self.vid_stride = vid_stride

        self.intergroup_index = 0  # 组间索引
        self.intragroup_index = -1  # 组内索引。-1表示宫格显示，0-3表示单路显示

        self.scale = int(np.ceil(np.sqrt(group_scale)))  # 横纵方向的视频数量

        self.im_show = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)

        self.run = True  # 运行标志
        self.first_run = True  # 第一次运行标志

        self.splicer = SquareSplice(self.scale, show_shape=(self.show_w, self.show_h))

    def start(self):
        if self.first_run == True:  # 第一次运行
            self.first_run = False
            source_list = Path(self.source).read_text().rsplit()
            n = len(source_list)
            self.groups_num = int(np.ceil(n/self.group_scale))  # 组数
            self.tracker_thread_list = []
            # self.q_in_list = [Queue(30)] * n
            self.q_in_list = []
            # self.q_out_list = [Queue(30)] * n

            # 追踪检测线程
            for i, source in enumerate(source_list):
                q_in = Queue(30)
                self.q_in_list.append(q_in)
                # q_out = Queue(30)
                tracker_thread = threading.Thread(
                    target=self.run_tracker_in_thread,
                    args=( self.weight, self.imgsz, source, self.vid_stride, i, q_in),
                    daemon=False
                    )
                self.tracker_thread_list.append(tracker_thread)
                tracker_thread.start()

            # 更新结果线程
            show_thread = threading.Thread(target=self.update_results, daemon=False)
            show_thread.start()

    def stop(self):
        self.run = False
        for tracker_thread in self.tracker_thread_list:
            tracker_thread.join()
        for q in self.q_in_list:
            q.queue.clear()
        print('模型已结束')
        self.run = True
        self.first_run = True

    def join(self):
        for tracker_thread in self.tracker_thread_list:
            tracker_thread.join()

    def next_group(self):
        self.intergroup_index = (self.intergroup_index + 1) % self.groups_num
        return f'当前显示第{self.intergroup_index}组视频'

    def prior_group(self):
        self.intergroup_index = (self.intergroup_index - 1) % self.groups_num
        return f'当前显示第{self.intergroup_index}组视频'
    
    # 选择组内视频
    def select_intragroup(self, d_click_rate: tuple):
        x, y = d_click_rate
        self.intragroup_index = int(x//(1/self.scale) + (y//(1/self.scale))*self.scale)
        return f'当前显示第{self.intergroup_index}组，第{self.intragroup_index}路视频'

    def exit_intragroup(self):
        self.intragroup_index = -1
        return f'当前显示第{self.intergroup_index}组视频'

    def get_results(self):
        return self.im_show

    def update_results(self):
        # temp_grid = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)
        # temp_im = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        group = [None] * self.group_scale
        result_groups = [None] * self.groups_num
        avg_fps = 0
        while self.run:
            t1 = time.time()
            for i, q in enumerate(self.q_in_list):
                group[i%self.group_scale] = q.get()

                if i%self.group_scale == self.group_scale-1:  # 一组视频收集完毕
                    if self.intragroup_index == -1:  # 宫格显示
                        result_groups[i//self.group_scale] = self.splicer(group)  # 拼接图片
                    else:  # 单路显示
                        result_groups[i//self.group_scale] = group[self.intragroup_index]

            self.im_show = result_groups[self.intergroup_index]
            self.im_show = cv2.putText(self.im_show, f"FPS={avg_fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 显示fps
            avg_fps = (avg_fps + (self.vid_stride / (time.time() - t1))) / 2
        print('collect_results结束')
    
    def read_frames(sources, q_out_list):
        pass

    # 需要抽象为类，每路加载不同的配置文件
    def run_tracker_in_thread(self, weight, imgsz, stream, vid_stride, file_index, q):
        """
        Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

        This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
        tracking. The function runs in its own thread for concurrent processing.

        Args:
            filename (str): The path to the video file or the identifier for the webcam/external camera source.
            weight (str): The YOLO weight path.
            file_index (int): An index to uniquely identify the file being processed, used for display purposes.

        Note:
            Press 'q' to quit the video display window.
        """
        tracker = Track(weight, imgsz, vid_stride=vid_stride)
        # warmup
        _, _ = tracker(self.im_show, {})

        # opencv-python                4.8.1.78
        cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)  # Read the video file
        while self.run:
            # print(f'第{file_index}路:{self.run}')
            success, frame = cap.read()  # Read the video frames
            if not success:
                cap.release()
                cap = cv2.VideoCapture(stream)
            
            annotated_frame, show_id = tracker(frame, {})
            q.put(annotated_frame) if not q.full() else q.get()
        # Release video sources
        cap.release()
        print(f"第{file_index}路已停止")


if __name__ == "__main__":
    weight = r'E:\Projects\weight\yolo\v8\detect\coco\yolov8s.pt'
    stream = 'list.streams'
    imgsz = 640
    predicter = SmartBackend(weight, stream, imgsz)
    predicter.start()
    print('done!')


    # model = YOLO(weight, task='detect')
    # results = model.track(
    #     source=stream,
    #     classes=[0, 2],
    #     # tracker="bytetrack.yaml",  # 20fps
    #     imgsz=imgsz,
    #     stream=True,
    #     # show=True,
    #     # verbose=False
    #     )  # 生成器

    # for result in results:
    #     print(result)
