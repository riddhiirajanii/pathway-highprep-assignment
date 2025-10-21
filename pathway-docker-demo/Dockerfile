# Use a slim Linux Python image
FROM python:3.11-slim

# avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# install system deps needed by pathway (Rust engine libs). For many cases no extra apt packages needed,
# but it's common to add build-essential / libssl if pip wheels need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage docker cache
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app and data
COPY . /app

# Create output directory
RUN mkdir -p /app/out

# Use a non-root user for safety (optional)
# RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# Default command: run the app
CMD ["python", "app.py"]
