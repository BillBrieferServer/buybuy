FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn jinja2 python-multipart \
    python-dotenv anthropic psycopg2-binary itsdangerous markdown

COPY app/ ./app/
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
