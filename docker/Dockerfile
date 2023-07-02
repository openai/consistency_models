FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libopenmpi-dev git wget \
    python3 python3-dev python3-pip python3-setuptools python3-wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging mpi4py \
    && pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install flash-attn==0.2.8

WORKDIR /home/
RUN pip3 install -e git+https://github.com/openai/consistency_models.git@main#egg=consistency_models \
    && ln -s /usr/bin/python3 /usr/bin/python
