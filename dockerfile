# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install the CPU build of PyTorch (same version as requirements.txt), avoiding the CUDA libraries
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

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
