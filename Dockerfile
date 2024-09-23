# Use the official Python image from the slim variant to reduce image size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENV=production

# Create working directory
WORKDIR /app

# Install system dependencies for psycopg2 and torch
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements to install dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Specify the entrypoint for running the application with Gunicorn
CMD exec gunicorn --bind :8080 --workers 4 --threads 8 --timeout 0 app:app
