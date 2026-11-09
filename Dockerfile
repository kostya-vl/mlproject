# syntax=docker/dockerfile:1.7
# Базовый образ: python 3.11 (Debian 12 bookworm), digest из репозитория (ЛР3/ЛР4).
FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ARROW_USER_SIMD_LEVEL=NONE \
    MODELS_DIR=/app/models

# non-root пользователь (UID/GID 10001)
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# runtime-зависимости: ML + API (без dev-инструментов)
COPY requirements-runtime.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt -r requirements-api.txt

# только то, что нужно сервису
COPY src/__init__.py src/predictor.py src/preprocessing.py src/metrics.py ./src/
COPY app/ ./app/
COPY static/ ./static/

# модель + SHA256SUMS — явный список (контроль включения)
COPY models/lab2_span_tfidf.joblib  ./models/
COPY models/lab2_span_logreg.joblib ./models/
COPY models/SHA256SUMS              ./models/

USER 10001:10001

# healthcheck: API отвечает healthy (модель загружена, SHA-256 совпал)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import json,urllib.request,sys; \
r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); \
sys.exit(0 if r.status==200 and json.load(r).get('status')=='healthy' else 1)"

# внутри контейнера — все интерфейсы; публикация на хосте — только loopback
EXPOSE 8000

# API_TOKEN и METRICS_TOKEN — ТОЛЬКО через runtime env (--env-file/-e), не в образе
ENTRYPOINT ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
