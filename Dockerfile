# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY README.md .

# Install uv
RUN pip install --no-cache-dir uv

# Install project dependencies
RUN uv pip install --system --no-cache -e .

# Create output directories
RUN mkdir -p outputs \
    src/mlproject/data/processed \
    src/mlproject/data/features

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command (interactive bash)
CMD ["/bin/bash"]
