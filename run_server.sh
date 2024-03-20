git clone https://github.com/wgh-z/algorithm_server.git
cd algorithm_server

docker build --pull --rm -f "Dockerfile.algorithm_server" -t algorithm_server:2.2.0-cuda12.1-cudnn8-runtime "."

docker run -d --gpus all --network=host --name a_server algorithm_server:2.2.0-cuda12.1-cudnn8-runtime

# docker run --rm -it --gpus all -p 8554:8554 -p 4006:4006 --name a_server algorithm_server:2.2.0-cuda12.1-cudnn8-runtime