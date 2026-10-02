# Simplatab image. Two variants, built from this file:
#   CPU (default):  docker build -t simplatab .
#   GPU (NVIDIA):   docker build --build-arg DEVICE=gpu -t simplatab:gpu .
#                   docker run --gpus all --shm-size=4g -p 7111:5000 simplatab:gpu
# The GPU variant uses the CUDA 12.8 build of PyTorch (bundled in its wheels: the host only needs
# the NVIDIA driver and the NVIDIA Container Toolkit) and is built for linux/amd64.

# Use an official Python runtime as a parent image
FROM python:3.9-slim

LABEL org.opencontainers.image.title="simplatab-machine-learning-automator"

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# The image's pip (23.0) rejects wheels of the PyTorch index whose metadata name differs in case
# (e.g. Jinja2) and then fails building them from source: use a recent pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch and torchvision (same versions as requirements.txt).
# DEVICE=gpu: the PyPI wheels, which bundle CUDA 12.8 (linux/amd64).
# DEVICE=cpu on linux/amd64 (Linux, Windows, Intel Macs): the CPU index, avoiding the CUDA libraries.
# DEVICE=cpu on linux/arm64 (Apple Silicon Macs, ARM Linux): the PyPI wheels are already CPU-only.
# TARGETARCH is set by BuildKit for the platform being built.
ARG DEVICE=cpu
ARG TARGETARCH
RUN if [ "$DEVICE" = "gpu" ] || [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0; \
    else \
        pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install Flask and other necessary packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the pretrained TabPFNv2 and TabICL weights into the image (otherwise fetched on first use)
COPY Helpers/dl_classifiers.py Helpers/dl_classifiers.py
RUN python Helpers/dl_classifiers.py || echo "WARNING: pretrained weights not downloaded, they will be downloaded on first use"

# Download the pretrained weights of the 10 networks of the image automator
COPY Helpers/image/models.py Helpers/image/models.py
RUN python Helpers/image/models.py || echo "WARNING: pretrained weights not downloaded, they will be downloaded on first use"

# Download the COCO-pretrained weights of the 10 detectors of the object detection automator
COPY Helpers/detection/models.py Helpers/detection/models.py
RUN python Helpers/detection/models.py || echo "WARNING: pretrained weights not downloaded, they will be downloaded on first use"

# Download the pretrained weights of the 3D networks (MedicalNet, Kinetics video networks, SwinUNETR
# self-supervised encoder) of the 3D image classification
COPY Helpers/image3d/models.py Helpers/image3d/models.py
RUN python Helpers/image3d/models.py || echo "WARNING: pretrained weights not downloaded, they will be downloaded on first use"

# Download the ImageNet weights of the pretrained 2D encoders of the image segmentation
COPY Helpers/segmentation/models.py Helpers/segmentation/models.py
RUN python Helpers/segmentation/models.py || echo "WARNING: pretrained weights not downloaded, they will be downloaded on first use"

# Copy the current directory contents into the container
COPY . .

# Create Materials directory inside the container
RUN mkdir -p ./Materials

# Version of the release, set by the CI (same as the image and release tags); shown in the web interface.
# Declared last so that a new version does not invalidate the cached dependency layers.
ARG APP_VERSION=dev
ENV SIMPLATAB_VERSION=$APP_VERSION
LABEL org.opencontainers.image.version=$APP_VERSION

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
