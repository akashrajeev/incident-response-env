# Root-level Dockerfile for Hugging Face Docker Spaces (default build path).
# Identical to server/Dockerfile; keep in sync when changing dependencies.
FROM python:3.11-slim

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
