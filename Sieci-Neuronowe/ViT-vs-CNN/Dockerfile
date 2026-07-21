# Dockerfile dla GRUPA 3 - Vision Transformer vs CNN Analysis
FROM python:3.9-slim

LABEL maintainer="Project: Vision Transformer vs CNN"
LABEL description="Jupyter environment for analyzing ViT vs CNN results"

WORKDIR /app

# Zainstaluj wymagane pakiety
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Zainstaluj Python pakiety
RUN pip install --no-cache-dir \
    jupyter==1.0.0 \
    jupyterlab==3.6.4 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.0

# Skonfiguruj Jupyter
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

# Domyślna komenda - uruchom Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
