# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies for Rust, FAISS, and Python build
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    curl \
    pkg-config \
    git \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && export PATH="$HOME/.cargo/bin:$PATH" \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Expose the port
EXPOSE 8000

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
