# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . .

# Install Flask and other necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Create Materials directory inside the container
RUN mkdir -p ./Materials

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]