# Use a standard Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies for building packages
RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Also install uvicorn for the server
RUN pip install --no-cache-dir "uvicorn[standard]" "fastapi"

# Copy all the necessary application code
COPY ./src ./src
COPY ./pipelines ./pipelines
COPY ./config ./config
COPY ./data ./data
COPY ./main.py ./main.py

# Set the Python path to include the app root and src
ENV PYTHONPATH=/app:/app/src

# Expose the port the server will run on
EXPOSE 9099

# The command to run the application
CMD ["python", "-u", "/app/main.py"]