# Use a base image with Python
FROM python:3.11.2

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run your application
CMD ["python3", "crop_detect.py"]
