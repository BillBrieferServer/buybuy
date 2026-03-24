FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi==0.135.2 uvicorn==0.42.0 jinja2 python-multipart \
    python-dotenv anthropic==0.86.0 psycopg2-binary itsdangerous markdown

COPY app/ ./app/
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 8000

# Run as non-root user
RUN adduser --disabled-password --no-create-home --uid 1000 appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
