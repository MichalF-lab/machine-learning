FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1-mesa-glx libsm6 libxext6 libxrender1 \
        libgomp1 ffmpeg git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# opencv-headless first so albumentations doesn't pull in the GUI variant
RUN pip install --no-cache-dir \
    opencv-python-headless \
    albumentations \
    "torchmetrics[detection]" \
    tqdm

RUN python -c "import torch; print('[BUILD] torch', torch.__version__, '| CUDA', torch.version.cuda); \
               import torchvision; print('[BUILD] torchvision', torchvision.__version__)"

COPY App/ ./App/

CMD ["python", "App/train.py"]
