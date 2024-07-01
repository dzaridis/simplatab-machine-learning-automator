# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

RUN pip install virtualenv

# Create a virtual environment for Giskard and install Giskard
RUN virtualenv giskard_env
RUN . giskard_env/bin/activate && pip install giskard -U && pip install requests==2.25.1 urllib3==1.26.5

# Install Flask and other necessary packages
RUN pip install --no-cache-dir -r requirements.txt



# Define volumes
VOLUME ["/input_data", "/Materials"]
# Define the entrypoint to run the script
# Expose the port the app runs on
EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]