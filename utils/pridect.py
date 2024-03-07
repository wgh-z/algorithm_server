# 模型预测相关
import cv2
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.loaders import LoadStreams
from ultralytics.data.augment import LetterBox


# 需要抽象为类，每路加载不同的配置文件
def run_tracker_in_thread(filename, weight, file_index):
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
            verbose=False
            )
        res_plotted = results[0].plot()
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()


if __name__ == "__main__":
    weight = r'E:\Projects\weight\yolo\v8\detect\coco\yolov8m.pt'
    stream = 'algorithm_server/list.streams'
    imgsz = 640

    source_list = Path(stream).read_text().rsplit()
    tracker_thread_list = []

    for i, source in enumerate(source_list):
        tracker_thread = threading.Thread(target=run_tracker_in_thread, args=(source, weight, i), daemon=True)
        tracker_thread_list.append(tracker_thread)
        tracker_thread.start()

    for tracker_thread in tracker_thread_list:
        tracker_thread.join()

    # Clean up and close windows
    cv2.destroyAllWindows()


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
