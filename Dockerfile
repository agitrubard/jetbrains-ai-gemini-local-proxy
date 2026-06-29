FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 5003

# --http h11 (NOT the default httptools): JetBrains' ktor-client sends an
# HTTP/1.1 -> HTTP/2 cleartext upgrade ("Connection: Upgrade", "Upgrade: h2c").
# uvicorn's httptools parser mishandles this non-websocket upgrade and DROPS the
# request body, so every real IDE request arrives empty -> 400. The h11 parser
# ignores the unsupported upgrade per spec and delivers the body normally.
# --no-access-log keeps the hot path lean for streaming; the app emits its own
# request/diagnostic logs (level controlled by the LOG_LEVEL env var).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5003", "--http", "h11", "--no-access-log"]
