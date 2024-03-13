# 自定义模型预测相关
import cv2
import time
import threading
import numpy as np
# import torch
from pathlib import Path
from queue import Queue

# from ultralytics import YOLO
# from ultralytics.data.loaders import LoadStreams
# from ultralytics.data.augment import LetterBox

from utils.toolbox import SquareSplice
from utils.video_io import VideoDisplayManage, ReadVideo
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
        self.source_list = Path(source).read_text().rsplit()
        self.imgsz = imgsz
        self.show_w = show_w
        self.show_h = show_h
        self.vid_stride = vid_stride

        # 参数计算
        self.n = len(self.source_list)
        self.group_scale = group_scale
        self.scale = int(np.ceil(np.sqrt(group_scale)))  # 横纵方向的视频数量
        self.groups_num = int(np.ceil(self.n / self.group_scale))  # 组数

        self.running = False  # 运行标志

        # 初始化共享变量
        self.im_show = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        self.video_reader_list = [None] * self.n
        self.tracker_thread_list = [None] * self.n
        self.q_in_list = [Queue(30) for _ in range(self.n)]
        # self.q_out_list = [Queue(30) for _ in range(n)]
        self.frame_list = [None] * self.n  # 用于存储每路视频的帧

        # 工具类
        self.splicer = SquareSplice(self.scale, show_shape=(self.show_w, self.show_h))
        self.display_manager = VideoDisplayManage(self.groups_num, self.scale)

    def start(self):
        if not self.running:  # 防止重复启动
            self.running = True

            # 追踪检测线程
            for i, source in enumerate(self.source_list):
                self.video_reader_list[i] = ReadVideo(source)

                tracker_thread = threading.Thread(
                    target=self.run_in_thread,
                    args=(self.weight, self.imgsz, self.vid_stride, i,
                          self.q_in_list[i], self.video_reader_list[i]),
                    daemon=False
                )
                self.tracker_thread_list[i] = tracker_thread
                tracker_thread.start()

            # 更新结果线程
            show_thread = threading.Thread(target=self.update_results, daemon=False)
            show_thread.start()

    def stop(self):
        self.im_show = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        self.running = False
        self.wait_thread()
        self.clear_up()
        print('模型已结束')

    def wait_thread(self):
        for tracker_thread in self.tracker_thread_list:
            tracker_thread.join()

    def clear_up(self):
        for video_reader in self.video_reader_list:
            video_reader.release()

        for q in self.q_in_list:
            q.queue.clear()

        # 清理显示管理器
        self.display_manager.reset()

        # 清理显存
        # if torch.cuda.is_available():
        #     with torch.cuda.device('cuda:0'):
        #         torch.cuda.empty_cache()
        #         torch.cuda.ipc_collect()

    def get_results(self):
        return self.im_show

    def update_results(self):
        avg_fps = 0
        wait_time = 1 / (25*self.n)  # 降低cpu占用
        while self.running:
            start_time = time.time()
            # group = [None] * self.group_scale
            for i, q in enumerate(self.q_in_list):
                if not q.empty():  # 异步获取结果，防止忙等待
                    self.frame_list[i] = q.get()
                else:
                    time.sleep(wait_time/2)
                    if not q.empty():
                        self.frame_list[i] = q.get()

            if self.display_manager.intragroup_index == -1:  # 宫格显示
                    start = self.display_manager.intergroup_index * self.group_scale
                    self.im_show = self.splicer(self.frame_list[start:start+self.group_scale])
                # group = [None] * self.group_scale
            else:  # 单路显示
                self.im_show = self.frame_list[self.display_manager.intragroup_index]

            self.im_show = cv2.putText(self.im_show, f"FPS={avg_fps:.2f}", (0, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 显示fps

            self.im_show = cv2.putText(self.im_show, f"{self.display_manager.get_display_index()}",
                                       (self.show_w-100, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # 显示组数和路数

            end_time = time.time()
            if end_time - start_time < wait_time:  # 帧数稳定
                time.sleep(wait_time - (end_time - start_time))
                print('显示空等待')
            avg_fps = (avg_fps + (1 / (time.time() - start_time))) / 2
        print('collect_results结束')

    def update_results2(self):
        group = [None] * self.group_scale
        result_groups = [None] * self.groups_num
        avg_fps = 0
        while self.running:
            t1 = time.time()
            for i, q in enumerate(self.q_in_list):
                group[i % self.group_scale] = q.get()

                if i % self.group_scale == self.group_scale - 1:  # 一组视频收集完毕
                    if self.display_manager.intragroup_index == -1:  # 宫格显示
                        result_groups[i // self.group_scale] = self.splicer(group)  # 拼接图片
                    else:  # 单路显示
                        result_groups[i // self.group_scale] = group[self.display_manager.intragroup_index]

            self.im_show = result_groups[self.display_manager.intergroup_index]
            self.im_show = cv2.putText(self.im_show, f"FPS={avg_fps:.2f}", (0, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 显示fps

            self.im_show = cv2.putText(self.im_show, f"{self.display_manager.get_display_index()}",
                                       (self.show_w-100, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # 显示组数和路数
            avg_fps = (avg_fps + (self.vid_stride / (time.time() - t1))) / 2
        print('collect_results结束')

    def read_frames(self, sources, q_out_list):
        pass

    # 需要抽象为类，每路加载不同的配置文件
    def run_in_thread(self, weight, imgsz, vid_stride,
                      index, q, video_reader):
        """
        """
        tracker = Track(weight, imgsz, vid_stride=vid_stride)
        _, _ = tracker(self.im_show, {})  # warmup

        wait_time = 1 / 25
        while self.running:
            t1 = time.time()
            # print(f'第{file_index}路:{self.run}')
            frame = video_reader()
            if frame is None:
                break

            annotated_frame, show_id = tracker(frame, {})
            t2 = time.time()
            if t2 - t1 < wait_time:  # 帧数稳定
                time.sleep(wait_time - (t2 - t1))
                print('检测空等待')
            q.put(annotated_frame) if not q.full() else q.get()
        print(f"第{index}路已停止")
 