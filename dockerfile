# Use an official Python runtime as a parent image
FROM python:3.9-slim

LABEL org.opencontainers.image.title="simplatab-machine-learning-automator" \
      org.opencontainers.image.version="1.1.0"

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# The image's pip (23.0) rejects wheels of the PyTorch index whose metadata name differs in case
# (e.g. Jinja2) and then fails building them from source: use a recent pip
RUN pip install --no-cache-dir --upgrade pip

# Install the CPU build of PyTorch (same version as requirements.txt), avoiding the CUDA libraries.
# linux/amd64 (Linux, Windows, Intel Macs): the PyPI wheel bundles CUDA, use the CPU index.
# linux/arm64 (Apple Silicon Macs, ARM Linux): the PyPI wheel is already CPU-only.
# TARGETARCH is set by BuildKit for the platform being built.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir torch==2.8.0; \
    else \
        pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install Flask and other necessary packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the pretrained TabPFNv2 and TabICL weights into the image (otherwise fetched on first use)
COPY Helpers/dl_classifiers.py Helpers/dl_classifiers.py
RUN python Helpers/dl_classifiers.py || echo "WARNING: pretrained weights not downloaded, they will be downloaded on first use"

# Copy the current directory contents into the container
COPY . .

# Create Materials directory inside the container
RUN mkdir -p ./Materials

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
