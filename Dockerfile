# Use official slim Python
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal system deps for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application and deps
COPY legendary_empire_app.py /app/legendary_empire_app.py
COPY requirements.txt /app/requirements.txt

# Install python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Default to running configuration tests; override CMD in runtime to run bot/dashboard
CMD ["python", "legendary_empire_app.py", "test-config"]
