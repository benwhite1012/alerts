# syntax=docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# ffmpeg for whisper
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache Whisper model so it doesn't download at runtime
ENV WHISPER_MODEL=base
RUN python - <<'PY'
import os, whisper
whisper.load_model(os.environ.get("WHISPER_MODEL","base"))
print("Cached whisper model.")
PY

# Copy app code
COPY . .

CMD ["python", "app.py"]
