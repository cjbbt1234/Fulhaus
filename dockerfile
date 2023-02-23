# Use PyTorch as the base image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Copy files from current directory to the /app directory in the container
COPY . /app

# Install Flask and Pillow
RUN pip install Flask Pillow

# Set working directory to /app
WORKDIR /app

# Map port 5000 of the container to port 5000 of the host
EXPOSE 5000

# Start the Flask application
CMD ["python", "app.py"]