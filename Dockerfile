# HELP: This Dockerfile is used to build the PlaneRecTR environment
# HELP: The Dockerfile is based on the PyTorch image with CUDA 11.3 and cuDNN 8
# HELP: The Dockerfile installs the required dependencies for the PlaneRecTR model
# HELP: The Dockerfile also installs the Detectron2 library and builds the custom CUDA ops for the PlaneRecTR model
# Steps:
# 1. git clone git@github.com:SJingjia/PlaneRecTR.git
# 2. cd PlaneRecTR
# 3. docker build -t planerectr .
# 4. docker run -it --gpus all -v "$(pwd)/checkpoint:/workspace/checkpoint" planerectr $CMD

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# We are not able to update the NVIDIA libs anymore hence removed the source files
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \ 
    libglib2.0-0 \
    git && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST="8.6"


WORKDIR /workspace
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
ADD requirements.txt /workspace
RUN pip install -r requirements.txt

RUN git clone -b v0.6 http://github.com/facebookresearch/detectron2.git
WORKDIR /workspace/detectron2
RUN pip install -e .

ADD PlaneRecTR /workspace/PlaneRecTR
WORKDIR /workspace/PlaneRecTR/modeling/pixel_decoder/ops
ENV FORCE_CUDA="1"
RUN sh make.sh

WORKDIR /workspace
ADD . /workspace

CMD bash
