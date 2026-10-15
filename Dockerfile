# syntax=docker/dockerfile:1

# Stage 1: training dependencies, code, and data (no model yet).
# docker-compose reuses this stage to retrain into a shared volume.
FROM python:3.11-slim AS train-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt
COPY src/train.py .
COPY data/telco_churn.csv data/telco_churn.csv

# Stage 2: train the model at build time and keep only the artifacts.
FROM train-base AS trainer
RUN python train.py --data data/telco_churn.csv --out /artifacts

# Stage 3: serving runtime with no model baked in.
# docker-compose uses this stage and loads the model from the shared volume.
FROM python:3.11-slim AS serve-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY src/app.py .
COPY src/templates templates
COPY src/statics statics
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4 (default target): serving runtime plus the model from stage 2.
FROM serve-base AS serving
COPY --from=trainer --chown=appuser:appuser /artifacts /app/artifacts
