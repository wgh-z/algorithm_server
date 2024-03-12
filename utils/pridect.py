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
            group_scale: int = 4,
            show_w: int = 1920,
            show_h: int = 1080
            ) -> None:
        self.weight = weight
        self.source = source
        self.imgsz = imgsz
        self.group_scale = group_scale

        self.group_num = 0

        self.scale = int(np.ceil(np.sqrt(group_scale)))
        self.grid_w = int(show_w / self.scale)
        self.grid_h = int(show_h / self.scale)

        self.im_show = np.zeros((show_h, show_w, 3), dtype=np.uint8)

    def run(self):
        source_list = Path(self.source).read_text().rsplit()
        self.tracker_thread_list = []
        self.queue_list = []
        for i, source in enumerate(source_list):
            q_in = Queue(30)
            tracker_thread = threading.Thread(target=self.run_tracker_in_thread, args=(source, self.weight, i, q_in), daemon=False)
            self.queue_list.append(q_in)
            self.tracker_thread_list.append(tracker_thread)
            tracker_thread.start()
        
        # show_thread = threading.Thread(target=self.show_results, daemon=False)
        # show_thread.start()

        # for tracker_thread in self.tracker_thread_list:
        #     tracker_thread.join()

        # Clean up and close windows
        # cv2.destroyAllWindows()

    def join(self):
        for tracker_thread in self.tracker_thread_list:
            tracker_thread.join()
        
    def next_group(self):
        self.group_num = (self.group_num + 1) % self.group_scale
        return self.group_num
    
    def prior_group(self):
        self.group_num = (self.group_num - 1) % self.group_scale
        return self.group_num

    def show_results(self):
        while True:
            try:
                t1 = time.time()
                frame = self.collect_results()[self.group_num]
                cv2.imshow('result', frame)
                cv2.waitKey(1)
                print('fps:', 1 / (time.time() - t1))
            except:
                pass
    
    def collect_results(self):
        group = []
        result_groups = []
        for q in self.queue_list:
            group.append(q.get())

            if len(group) == self.group_scale:
                for i, im0 in enumerate(group):  # 拼接
                    im0 = cv2.resize(im0, (self.grid_w, self.grid_h))
                    self.im_show[self.grid_h*(i//self.scale):self.grid_h*(1+(i//self.scale)),
                            self.grid_w*(i%self.scale):self.grid_w*(1+(i%self.scale))] = im0
                show_img = self.im_show.copy()
                result_groups.append(show_img)
                group = []
        return result_groups[self.group_num]


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
                classes=[0,2],
                tracker="bytetrack.yaml",  # 20fps
                persist=True,
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
    weight = r'E:\Projects\weight\yolo\v8\detect\coco\yolov8s.pt'
    stream = 'list.streams'
    imgsz = 640
    predicter = Pridect(weight, stream, imgsz)
    predicter.run()
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
