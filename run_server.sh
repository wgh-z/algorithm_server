#!/bin/bash

# chmod +x run_server.sh

# git clone https://github.com/wgh-z/algorithm_server.git
# cd algorithm_server

# docker build --pull --rm -f "Dockerfile.algorithm_server" -t algorithm_server:2.2.0-cuda12.1-cudnn8-runtime "."

# docker run -d --gpus all --network=host --name a_server algorithm_server:2.2.0-cuda12.1-cudnn8-runtime

PORT=${1:-8554}

sudo docker run --rm -it --gpus all \
    -p $PORT:8554 -p 4006:4006 \
    -v /home/wgh-ubuntu/algorithm_server:/usr/src/ultralytics \
    --name a_server algorithm_server:2.2.0-cuda12.1-cudnn8-runtime