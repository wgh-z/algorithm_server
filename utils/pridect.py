# 模型预测相关
import cv2
import time
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.loaders import LoadStreams
from ultralytics.data.augment import LetterBox
from queue import Queue


class Pridect:
    def __init__(
            self,
            weight,
            source,
            imgsz,
            group_scale: int = 4,  # 每组4路视频
            show_w: int = 1920,
            show_h: int = 1080
            ) -> None:
        self.weight = weight
        self.source = source
        self.imgsz = imgsz
        self.group_scale = group_scale
        self.show_w = show_w
        self.show_h = show_h

        self.group_index = 0  # 当前显示的组索引

        self.scale = int(np.ceil(np.sqrt(group_scale)))  # 横纵方向的视频数量
        self.grid_w = int(self.show_w / self.scale)
        self.grid_h = int(self.show_h / self.scale)

        self.im_show = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)

        self.break_flag = False  # 停止标志
        self.first_run = True  # 第一次运行标志

    def start(self):
        if self.first_run == True:  # 第一次运行
            self.first_run = False
            source_list = Path(self.source).read_text().rsplit()
            self.group_num = int(np.ceil(len(source_list)/self.group_scale))  # 组数
            self.tracker_thread_list = []
            self.queue_list = []
            for i, source in enumerate(source_list):
                q_in = Queue(30)
                tracker_thread = threading.Thread(target=self.run_tracker_in_thread, args=(source, self.weight, i, q_in), daemon=True)
                self.queue_list.append(q_in)
                self.tracker_thread_list.append(tracker_thread)
                tracker_thread.start()
            
            show_thread = threading.Thread(target=self.collect_results, daemon=False)
            show_thread.start()

        # for tracker_thread in self.tracker_thread_list:
        #     tracker_thread.join()

        # Clean up and close windows
        # cv2.destroyAllWindows()
    
    def stop(self):
        self.break_flag = True
        # for tracker_thread in self.tracker_thread_list:
        #     tracker_thread.join()
        # cv2.destroyAllWindows()

    def join(self):
        for tracker_thread in self.tracker_thread_list:
            tracker_thread.join()
        
    def next_group(self):
        self.group_index = (self.group_index + 1) % self.group_scale
        return self.group_index
    
    def prior_group(self):
        self.group_index = (self.group_index - 1) % self.group_scale
        return self.group_index

    def show_results(self):
        while True:
            try:
                t1 = time.time()
                frame = self.collect_results()[self.group_index]
                cv2.imshow('result', frame)
                cv2.waitKey(1)
                print('fps:', 1 / (time.time() - t1))
            except:
                pass

    def get_results(self):
        return self.im_show
    
    def collect_results(self):
        # temp_grid = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)
        # temp_im = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)

        group = [None] * self.group_scale
        result_groups = [None] * self.group_num
        avg_fps = 0
        while True:
            print('collect_results', self.break_flag)
            if self.break_flag:
                break
            t1 = time.time()
            for i, q in enumerate(self.queue_list):
                group[i%self.group_scale] = q.get()

                if i%self.group_scale == self.group_scale-1:  # 一组视频收集完毕
                    result_groups[i//self.group_scale] = self.splice(group, self.scale)  # 拼接图片

            self.im_show = result_groups[self.group_index]
            self.im_show = cv2.putText(self.im_show, f"FPS={avg_fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 显示fps
            avg_fps = (avg_fps + (1 / (time.time() - t1))) / 2
    
    def splice(self, im_list, scale):
        """
        拼接图片
        """
        im = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        for i, im0 in enumerate(im_list):
            im0 = cv2.resize(im0, (self.grid_w, self.grid_h))
            im[self.grid_h*(i//scale):self.grid_h*(1+(i//scale)), self.grid_w*(i%scale):self.grid_w*(1+(i%scale))] = im0
        return im

    # 需要抽象为类，每路加载不同的配置文件
    def run_tracker_in_thread(self, filename, weight, file_index, q):
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
        model = YOLO(weight)
        video = cv2.VideoCapture(filename)  # Read the video file
        while True:
            ret, frame = video.read()  # Read the video frames
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                classes=[0,2],
                verbose=False
                )
            res_plotted = results[0].plot()
            # cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break

            q.put(res_plotted) if not q.full() else q.get()
        # Release video sources
        video.release()


if __name__ == "__main__":
    weight = r'E:\Projects\weight\yolo\v8\detect\coco\yolov8m.pt'
    stream = 'list.streams'
    imgsz = 640
    predicter = Pridect(weight, stream, imgsz)
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
