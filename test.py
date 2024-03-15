from utils.toolbox import create_model_config_flie


if __name__ == '__main__':
    streams = [
        'rtsp://192.168.31.181:8554/stream0',
        'rtsp://192.168.31.181:8554/stream1',
        'rtsp://192.168.31.181:8554/stream2',
        'rtsp://192.168.31.181:8554/stream3',
        'rtsp://192.168.31.181:8554/stream4',
        'rtsp://192.168.31.181:8554/stream5',
        'rtsp://192.168.31.181:8554/stream6',
        'rtsp://192.168.31.181:8554/stream7',
        'rtsp://192.168.31.181:8554/stream8',
        'rtsp://192.168.31.181:8554/stream9',
        'rtsp://192.168.31.181:8554/stream10',
        'rtsp://192.168.31.181:8554/stream11',
        'rtsp://192.168.31.181:8554/stream12',
        'rtsp://192.168.31.181:8554/stream13',
        'rtsp://192.168.31.181:8554/stream14',
        'rtsp://192.168.31.181:8554/stream15'
    ]
    cfg_dict = {
        'show': {
            'group_scale': None,
            'show_shape': None
        },
        'source':None
        }
    sources = {}
    for i, stream in enumerate(streams):
        sources[stream] = {
            'weight': 'yolov8s.pt',
            'task': 'track',
            'detect_size': [384, 640],
            'classes': [0, 2],
            'tracker': 'bytetrack.yaml',
            'group_num': i // 4,
            'video_stride': 8
        }
    print(cfg_dict)
    cfg_dict['source'] = sources

    # cfg_dict = {'source':{}}
    # for i, stream in enumerate(streams):
    #     cfg_dict['source'][stream] = None
    # print(cfg_dict)
    result = create_model_config_flie(cfg_dict, save_path='./cfg/track.yaml')
    print(result)
