FROM python:3.10-slim-bookworm

# cu126 is the newest PyTorch CUDA build that still ships Pascal (sm_61) kernels,
# so it runs on a GTX 10xx as well as newer cards. Use cu128/cu130 for Turing+ only.
ARG TORCH_CUDA=cu126
ARG TORCH_VERSION=
ARG DA3_REF=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
ARG BAKE_MODELS="da3-small da3-base da3-large"
ARG UID=1000
ARG GID=1000

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "torch${TORCH_VERSION:+==$TORCH_VERSION}" torchvision \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    && pip freeze | grep -E '^(torch|torchvision)==' > /tmp/torch-constraints.txt

# Weights before the other deps so requirement changes don't re-download them.
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app \
    && mkdir -p /opt/hf /hf-host /data/jobs && chown -R app:app /opt/hf /hf-host /data
COPY app/models.py /srv/app/models.py
RUN pip install huggingface_hub \
    && cd /srv && HF_HUB_CACHE=/opt/hf python -m app.models --bake ${BAKE_MODELS} \
    && chown -R app:app /opt/hf

COPY requirements.txt /tmp/requirements.txt
RUN pip install -c /tmp/torch-constraints.txt -r /tmp/requirements.txt \
    && pip install --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3@${DA3_REF}" \
    && python -c "from depth_anything_3.api import DepthAnything3"

COPY . /srv
WORKDIR /srv
USER app

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
