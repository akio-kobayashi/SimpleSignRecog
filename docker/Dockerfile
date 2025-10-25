# CUDA 12.6 + cuDNN + Ubuntu 24.04
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Tokyo

# OS依存パッケージのみインストールする
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    libgl1 libxext6 libsm6 libglib2.0-0 \
    build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# イメージ起動
CMD ["/bin/bash"]
