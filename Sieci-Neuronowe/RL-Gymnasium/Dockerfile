FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

LABEL description="GPU training env for Projekt_3 - MountainCar RL (DQN/REINFORCE/PPO)"

WORKDIR /app

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    gymnasium==1.3.0 \
    pandas==2.2.3 \
    matplotlib==3.9.2 \
    pygame==2.6.1

CMD ["python", "train_for_animation.py"]
