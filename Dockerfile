# Build stage: install deps and run import checks
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source and run import/compile tests to fail fast if any syntax/import error
COPY legendary_empire_app.py /app/legendary_empire_app.py
RUN python -m py_compile /app/legendary_empire_app.py

# Final runtime image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY legendary_empire_app.py /app/legendary_empire_app.py

CMD ["python", "legendary_empire_app.py", "test-config"]
