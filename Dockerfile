# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional if your libs need build tools)
RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config python3-dev

# Copy all files to container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --prefer-binary -r requirements.txt

# Expose the app port
EXPOSE 8000

# Command to run the app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
