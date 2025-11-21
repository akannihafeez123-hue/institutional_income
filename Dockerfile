FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY legendary_empire_app.py /app/

# Default command; override to run-bot or run-dashboard as needed
CMD ["python", "legendary_empire_app.py", "test-config"]
