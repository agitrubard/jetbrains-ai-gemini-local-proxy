FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 5003

# uvicorn[standard] auto-selects uvloop + httptools.
# --no-access-log keeps the hot path lean for streaming.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5003", "--no-access-log"]
