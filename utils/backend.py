# 自定义模型预测相关
import cv2
import time
import threading
import numpy as np
import yaml
from pathlib import Path
from queue import Queue

# from ultralytics import YOLO
# from ultralytics.data.loaders import LoadStreams
# from ultralytics.data.augment import LetterBox

from utils.toolbox import SquareSplice
from utils.video_io import VideoDisplayManage, ReadVideo
from models.track import Track


class SmartBackend:
    def __init__(self, config_path='./cfg/track.yaml',) -> None:
        self.config_path = config_path
        self.running = False  # 运行标志

    # config文件解析
    def parse_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # show键
        self.group_scale = cfg['show']['group_scale']  # 每组显示的视频数量
        self.show_w, self.show_h = cfg['show']['show_shape']  # 显示窗口大小
    
        # source键
        self.n = len(cfg['source'])  # 视频总数量
        self.source_dict = cfg['source']

        # 按组排序
        group_dict = {i: [] for i in range(self.group_scale)}
        for source in self.source_dict.keys():
            group_dict[self.source_dict[source]['group_num']].append(source)

        self.source_list = []
        for group_num in group_dict.keys():
            self.source_list += group_dict[group_num]

    def initialize(self):
        # 参数计算
        self.scale = int(np.ceil(np.sqrt(self.group_scale)))  # 横纵方向的视频数量
        self.groups_num = int(np.ceil(self.n / self.group_scale))  # 组数

        # 初始化共享变量
        self.im_show = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        self.video_reader_list = [None] * self.n
        self.tracker_thread_list = [None] * self.n
        self.q_in_list = [Queue(30) for _ in range(self.n)]
        self.q_out_list = [Queue(30) for _ in range(self.n)]
        self.frame_list = [None] * self.n  # 用于存储每路视频的帧

        # 工具类
        self.splicer = SquareSplice(self.scale, show_shape=(self.show_w, self.show_h))
        self.display_manager = VideoDisplayManage(self.group_scale, self.groups_num, self.scale)

    def start(self):
        if not self.running:  # 防止重复启动
            self.running = True
            self.parse_config()
            self.initialize()

            # 读取视频线程
            self.read_thread = threading.Thread(
                target=self.read_frames,                    
                args=(self.source_list, self.q_in_list),
                daemon=False
                )
            self.read_thread.start()

            # 检测线程
            self.pridector_thread = threading.Thread(
                target=self.run_in_thread,
                args=(
                    self.source_list,
                    self.source_dict,
                    self.q_in_list,
                    self.q_out_list
                    ),
                daemon=False
            )
            self.pridector_thread.start()

            # 更新结果线程
            self.show_thread = threading.Thread(target=self.update_results, daemon=False)
            self.show_thread.start()

    def stop(self):
        self.im_show = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        self.running = False
        self.wait_thread()
        self.clear_up()
        print('模型已结束')

    def wait_thread(self):
        if self.read_thread is not None:
            self.read_thread.join()
        if self.pridector_thread is not None:
            self.pridector_thread.join()
        if self.show_thread is not None:
            self.show_thread.join()

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
        wait_time = 1 / 30  # 降低cpu占用
        while self.running:
            start_time = time.time()
            # group = [None] * self.group_scale
            for i, q in enumerate(self.q_out_list):
                # if not q.empty():  # 异步获取结果，防止忙等待
                    self.frame_list[i] = q.get()
                # else:
                #     time.sleep(wait_time/self.n)
                #     print('等待结果')
                #     if not q.empty():
                #         self.frame_list[i] = q.get()

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
                # print('显示空等待')
            avg_fps = (avg_fps + (1 / (time.time() - start_time))) / 2
        print('collect_results结束')

    def read_frames(self, source_list: list[str], q_in_list: list[Queue]):
        # reader_list = []
        for i, source in enumerate(source_list):
            video_reader = ReadVideo(source)
            self.video_reader_list[i]=video_reader
        
        while self.running:
            for i, video_reader in enumerate(self.video_reader_list):
                frame = video_reader()
                if frame is None:
                    self.running = False
                    break
                if q_in_list[i].full():
                    q_in_list[i].get()
                    # print(f"第{i}路视频帧队列已满")
                q_in_list[i].put(frame)
        print(f"视频读取已停止")

    # 需要抽象为类，每路加载不同的配置文件
    def run_in_thread(self,
                      source_list: list[str],
                      source_dict: dict,
                      q_in_list: list[Queue],
                      q_out_list: list[Queue]
                      ):
        """
        """
        traker_list = []
        for i, source in enumerate(source_list):
            tracker = Track(
                weight=source_dict[source]['weight'],
                imgsz=source_dict[source]['detect_size'],
                classes=source_dict[source]['classes'],
                tracker=source_dict[source]['tracker'],
                vid_stride=source_dict[source]['video_stride'],
                index=i
                )
            _, _ = tracker(self.im_show, {})  # warmup
            traker_list.append(tracker)

        wait_time = 1 / 25
        while self.running:
            t1 = time.time()
            # print(f'第{index}路:{self.run}')
            detc = []
            for i, tracker in enumerate(traker_list):
                frame = q_in_list[i].get()
                annotated_frame, show_id = tracker(frame, {})
                # if q_out_list[i].full():
                #     q_out_list[i].get()
                q_out_list[i].put(annotated_frame)

                if tracker.detect == True:
                    detc.append(i)
            print(f'检测路数:{detc}')

            t2 = time.time()
            # if t3 - t1 < wait_time:  # 帧数稳定
            #     time.sleep(wait_time - (t2 - t1))
            #     print('检测空等待')
            # print(f'检测时间:{t2-t1}')  # 0.014,0.4/0.03
            # print(f'第{index}路检测:{tracker.count}')  # 0.014,0.291
        # print(f"第{index}路已停止")
 