FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    torchvision==0.19.0 \
    "ultralytics==8.3.0" \
    matplotlib==3.9.0 \
    pillow==10.4.0 \
    requests==2.32.0 \
    numpy==1.26.4 && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless==4.11.0.86

COPY analyze.py .

CMD ["python", "analyze.py"]
