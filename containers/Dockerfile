ARG NGC_TAG=24.10-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="70;80;90" HF_HOME=/workspace/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache MPLCONFIGDIR=/workspace/.cache/matplotlib
SHELL ["/bin/bash","-lc"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ninja-build cmake pkg-config wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY ../requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip wheel setuptools && \
    pip install -r /tmp/requirements.txt
ARG INSTALL_FLASH_ATTN=false
RUN if [ "$INSTALL_FLASH_ATTN" = "true" ]; then \
      pip install --no-build-isolation -v flash-attn; \
    fi
ENTRYPOINT []
CMD ["/bin/bash"]

