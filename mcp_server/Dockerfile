# Simplatab MCP server: every Simplatab automator as MCP tools for AI agents.
# Build from the repository root (the image copies the Simplatab pipelines from Helpers/):
#   CPU (default):  docker build -f mcp_server/Dockerfile -t simplatab-mcp .
#   GPU (NVIDIA):   docker build -f mcp_server/Dockerfile --build-arg DEVICE=gpu -t simplatab-mcp:gpu .
# Run (stdio, launched by an MCP client):
#   docker run -i --rm -v "$PWD/data:/data" -v simplatab-workspace:/workspace simplatab-mcp
# Run (Streamable HTTP on http://localhost:8000/mcp):
#   docker run --rm -p 8000:8000 -v "$PWD/data:/data" -v simplatab-workspace:/workspace simplatab-mcp \
#       --transport http --host 0.0.0.0 --port 8000
# GPU: add --gpus all --shm-size=4g to docker run (NVIDIA driver and Container Toolkit on the host).
#
# Two Python environments: the Simplatab pipelines run in Python 3.9 (the same stack as the web image,
# where every pipeline is tested); the MCP server runs in Python 3.12 (the MCP SDK needs >= 3.10) and
# starts the pipelines as worker processes.

FROM python:3.9-slim

LABEL org.opencontainers.image.title="simplatab-mcp-server"

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip

# PyTorch and torchvision (same versions as requirements.txt).
# DEVICE=gpu: the PyPI wheels, which bundle CUDA 12.8 (linux/amd64).
# DEVICE=cpu on linux/amd64: the CPU index; on linux/arm64 the PyPI wheels are already CPU-only.
ARG DEVICE=cpu
ARG TARGETARCH
RUN if [ "$DEVICE" = "gpu" ] || [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0; \
    else \
        pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

# The Simplatab pipelines and their dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pretrained weights of the automators, downloaded into the image (otherwise fetched on first use)
COPY Helpers/dl_classifiers.py Helpers/dl_classifiers.py
RUN python Helpers/dl_classifiers.py || echo "WARNING: TabPFNv2/TabICL weights not downloaded"
COPY Helpers/image/models.py Helpers/image/models.py
RUN python Helpers/image/models.py || echo "WARNING: image network weights not downloaded"
COPY Helpers/detection/models.py Helpers/detection/models.py
RUN python Helpers/detection/models.py || echo "WARNING: detector weights not downloaded"
COPY Helpers/image3d/models.py Helpers/image3d/models.py
RUN python Helpers/image3d/models.py || echo "WARNING: 3D network weights not downloaded"
COPY Helpers/segmentation/models.py Helpers/segmentation/models.py
RUN python Helpers/segmentation/models.py || echo "WARNING: segmentation encoder weights not downloaded"

# The MCP server's own Python (3.12) with the MCP SDK, installed by uv (from PyPI)
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
COPY mcp_server/requirements.txt mcp_server/requirements.txt
RUN pip install --no-cache-dir uv==0.8.17 \
    && uv venv --python 3.12 /opt/mcp \
    && uv pip install --no-cache --python /opt/mcp/bin/python -r mcp_server/requirements.txt

# Simplatab (pipelines and example data) and the MCP server
COPY Helpers/ Helpers/
COPY Examples/ Examples/
COPY mcp_server/simplatab_mcp/ simplatab_mcp/
RUN mkdir -p /workspace /data

ENV SIMPLATAB_ROOT=/app \
    SIMPLATAB_WORKSPACE=/workspace \
    SIMPLATAB_DATA_DIR=/data \
    SIMPLATAB_WORKER_PYTHON=/usr/local/bin/python \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

ARG APP_VERSION=dev
ENV SIMPLATAB_VERSION=$APP_VERSION
LABEL org.opencontainers.image.version=$APP_VERSION

VOLUME ["/workspace"]
EXPOSE 8000
ENTRYPOINT ["/opt/mcp/bin/python", "-m", "simplatab_mcp"]
