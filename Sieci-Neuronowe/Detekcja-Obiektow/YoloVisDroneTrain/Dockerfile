# Base: PyTorch 2.7.0 + CUDA 12.8 + cuDNN 9 — includes sm_120 (Blackwell) kernels
# Fallback tags if this one is unavailable:
#   pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
#   nvidia/cuda:12.8.1-cudnn9-runtime-ubuntu22.04  (then: pip install torch --index-url .../cu128)
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System libs required by OpenCV / ultralytics image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ffmpeg \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned ultralytics — pip won't downgrade torch 2.7 (ultralytics needs >=2.0.0)
RUN pip install --no-cache-dir ultralytics==8.4.48

# Build-time sanity check (no GPU here, just confirm torch/CUDA versions are correct)
RUN python -c "import torch; print('[BUILD] torch', torch.__version__, '| CUDA toolkit', torch.version.cuda)"

COPY App/ ./App/

CMD ["python", "App/TrainModel.py"]
