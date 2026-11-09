# syntax=docker/dockerfile:1.7
# Базовый образ: официальный python, зафиксированная версия.
# Для точной фиксации замените тег на digest:
#   docker pull python:3.11.9-slim-bookworm
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.11.9-slim-bookworm
#   FROM python:3.11.9-slim-bookworm@sha256:<digest>
FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ARROW_USER_SIMD_LEVEL=NONE

# Non-root пользователь (UID/GID 10001)
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Runtime-зависимости: только то, что нужно для инференса
COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Только необходимые модули: predictor + preprocessing + __init__
COPY src/__init__.py src/predictor.py src/preprocessing.py ./src/

# CLI для smoke-проверки
COPY scripts/predict_cli.py ./scripts/predict_cli.py
COPY static/ ./static/

# Модель: явный список файлов (контроль включения)
COPY models/lab2_span_tfidf.joblib  ./models/
COPY models/lab2_span_logreg.joblib ./models/
COPY models/SHA256SUMS              ./models/

USER 10001:10001

# Healthcheck: модель на месте и загружается (закрывает DS-0026).
# Для CLI-контейнера проверка лёгкая — без инференса.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.path.insert(0,'/app'); \
from pathlib import Path; import joblib; \
joblib.load('/app/models/lab2_span_tfidf.joblib'); \
joblib.load('/app/models/lab2_span_logreg.joblib'); \
print('healthy')" || exit 1

ENTRYPOINT ["python", "scripts/predict_cli.py"]
CMD ["ivan.petrov@example.com", "+7 900 123-45-678"]
