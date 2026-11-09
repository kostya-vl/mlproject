# ML Project: NLP для киберустойчивости (LR2)

Обнаружение и классификация персональных данных (ПДн) в текстах
согласно 152-ФЗ. Базовая модель — span-type classifier:
TF-IDF (char_wb n-граммы) + LogisticRegression.

## Структура проекта

```
.
├── data/                   # датасет, CHECKSUMS.sha256
├── notebooks/              # ноутбуки ЛР1/ЛР2
├── src/                    # проверяемые модули
│   ├── predictor.py        # валидация входа + предсказание
│   ├── train.py            # обучение модели (CLI)
│   └── __init__.py
├── models/                 # финальные артефакты + SHA256SUMS
├── artifacts/              # промежуточные артефакты
├── metrics/                # метрики и отчёты
├── tests/                  # pytest (11 негативных + 1 позитивный)
├── requirements.txt
├── .gitignore
└── README.md
```

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Данные

Датасет: scanpatch/pii-ner-corpus-synthetic-controlled (Hugging Face).

```bash
sha256sum -c data/CHECKSUMS.sha256
```

## Обучение

```bash
source venv/bin/activate
python -m src.train --artifacts artifacts --seed 42
```

Создаёт artifacts/lab2_span_tfidf.joblib, artifacts/lab2_span_logreg.joblib,
artifacts/lab2_model_report.json.

## Оценка на тесте

```bash
jupyter lab notebooks/lab2_metrics.ipynb
cat metrics/baseline.json
```

Основная метрика — macro_f1 на отложенной тестовой выборке.

## Тестирование (pytest)

```bash
source venv/bin/activate
python -m pytest tests/ -q
```

Ожидается: 12 passed. Проверяются: неверный тип, NaN/Inf, пустой вход,
лимиты, битый/подменённый файл. Для каждого негативного кейса — проверка,
что модель не вызывается.

## Предсказание

```python
import sys, os
sys.path.insert(0, os.path.expanduser('~/mlproject'))
from src.predictor import predict, predict_from_file

result = predict(['ivan.petrov@example.com', '+7 900 123-45-67'])
for r in result:
    print(r['label'], r['confidence'], r['span'])

result = predict_from_file('spans.txt')  # .txt, UTF-8, один спан на строку
```

## Статический анализ (Bandit)

```bash
source venv/bin/activate
python -m bandit -r src/
```

Отчёт: metrics/bandit_report.json.

## Артефакты модели

Финальные артефакты — models/:
- lab2_span_tfidf.joblib — TF-IDF-преобразование
- lab2_span_logreg.joblib — LogisticRegression

Проверка целостности:

```bash
cd models/
sha256sum -c SHA256SUMS
```

Манифест с хэшами и data_id — metrics/model_artifacts.json.
Параметры обучения — metrics/training_params.json.
Метрики — metrics/metrics_summary.json, metrics/baseline.json.

## Ограничения

- Данные синтетические: высокая accuracy на test не гарантирует
  обобщения на реальные ПДн.
- Модель типизирует готовые спаны; извлечение спанов (полный NER) —
  за рамками ЛР2.
- .joblib использует pickle: загружать только собственные артефакты.
- Jupyter запускать от mluser, слушать 127.0.0.1, доступ — через SSH-туннель.

## Политика сравнения кандидата с baseline (ЛР3, задание 6)

- **Метрика:** `macro_f1` (та же, что в `metrics/baseline.json` из ЛР2).
- **Порог:** `candidate >= baseline * (1 - 0.05)` — допустимое относительное
  ухудшение 5%. Если `baseline == 0`, порог — `absolute_min` из
  `configs/regression_policy.yaml`.
- **Baseline не перезаписывается:** `metrics/baseline.json` только читается.
  Результат текущего запуска — `metrics/candidate.json`,
  сравнение — `metrics/comparison.json`.
- **Отклоняются:** отсутствующая/нечисловая метрика, `NaN`/`Inf`,
  несовместимая версия данных (`data_version.snapshot_hash`,
  `data_version.test_arrow_sha256` должны совпадать с baseline).
- **Коды возврата `compare_models.py`:** `0` — pass, `1` — reject,
  `2` — инфраструктура (нет файла/конфига).
- **Демонстрация сбоя:** `make_weak_candidate.py` — константный предиктор
  (всегда самый частый класс train), честная оценка на том же `test.parquet`.
  Gate отклоняет его (`exit 1`). Это **оценка реальной слабой модели**,
  а не искусственный файл метрик.
- **Финальная независимая оценка:** CI-сравнение **не заменяет**
  финальную оценку на hold-out test — оно выполняется на той же
  `artifacts/test.parquet`, что и ЛР2, с той же предобработкой
  (`flatten_spans` + TF-IDF) и той же метрикой (`compute_metrics`).

## Пределы и защита входящих запросов (ЛР4)

| Предел | Значение | Обоснование |
|---|---|---|
| `MAX_BODY_BYTES` | 1 МБ | PII-спан короткий (email ≤254, телефон ≤20); 1000 спанов × ~100 символов ≈ 100 КБ, 1 МБ — 10× запас. Тело читается потоково (`app/middleware.py`), при превышении — 413 **до** полной загрузки в память; `Content-Length` — только быстрая отсечка, не единственная защита. |
| `MAX_SPANS_PER_REQUEST` | 1000 | sklearn-инференс CPU-bound; 1000 спанов ≈ 1–2 с. Пакет больше — через цикл с разбиением. |
| `MAX_SPAN_LEN_CHARS` | 1000 | PII-спан ≤ 200 символов на практике; 1000 — 5× запас. Длиннее — уже не спан. |
| `MAX_TOTAL_CHARS` | 1 000 000 | TF-IDF `char_wb` на 1 МБ — ~4 млн n-грамм, разреженная матрица помещается; выше — нелинейный рост памяти/латентности. |
| `MAX_CONCURRENT_REQUESTS` | 4 | CPU-bound sklearn; 4 = типичное число vCPU. `asyncio.Semaphore` ограничивает параллелизм, а не только «принимает» запросы. |
| `REQUEST_TIMEOUT_SEC` | 10 с | Базовая модель на 1000 спанах — <1 с; 10 с — потолок. По таймауту `asyncio.wait_for` отменяет ожидание, поток освобождается; клиент получает 504. Клиентский timeout **не** останавливает серверное вычисление — поэтому ограничение на сервере. |

**Коды ошибок:** 400 (malformed JSON), 413 (body > 1 МБ), 415 (`content-type` ≠ `application/json`), 422 (схема/типы/NaN/Inf/лимиты спанов), 503 (модель не готова), 504 (timeout). Тело ответа — `{"detail": ...}` без traceback и без путей.

**Файлы пользователя:** сервис принимает только JSON `{"spans": [...]}`; загрузка файлов и артефактов модели пользователем **не предусмотрена**. Имя файла/путь из запроса нигде не используется для записи. Модель загружается только из `models/` при старте, SHA-256 сверяется с `models/SHA256SUMS`.

**Reverse proxy:** в проде перед сервисом ставится nginx с `client_max_body_size 1m` — это дополнительная отсечка до чтения тела в приложении. В ЛР4 middleware в `app/middleware.py` выполняет ту же роль без внешнего прокси.

## Аутентификация и rate limiting (ЛР4, задание 4)

### API-токен

- `/predict` защищён заголовком `X-API-Key`. Секрет хранится **только** в env
  `API_TOKEN` (в Git/Dockerfile/образе его нет). Если `API_TOKEN` не задан —
  сервис возвращает 503 `auth not configured`, а не пускает «по умолчанию».
- Отсутствующий или неверный токен → **401** (`WWW-Authenticate: ApiKey`).
- Роли не реализованы → **403** не возвращается. При добавлении ролей:
  403 для аутентифицированного клиента без прав на операцию.
- Токен не логируется и не возвращается в ответе. В логах — только `request_id`.

### Rate limiting

- Лимит: **30 запросов / 60 секунд** на ключ (`RATE_LIMIT_MAX`/`RATE_LIMIT_WINDOW`).
- Ключ ограничения: `sha256(токен)[:16] + IP клиента` — токен в памяти/логах
  в открытом виде не хранится.
- При превышении → **429** + `Retry-After`.
- `X-Forwarded-For` учитывается **только** если запрос пришёл от
  `TRUSTED_PROXY_IP` (задаётся env); иначе используется `request.client.host`.
  Это защита от подмены forwarded-заголовка клиентом.
- Реализация — in-memory fixed window (один процесс). **Ограничения:**
  при N workers фактический лимит ≈ N × limit; при перезапуске состояние
  сбрасывается; для нескольких инстансов нужен общий store (Redis) или
  rate limiting на reverse proxy (nginx `limit_req`).

### Демо-страница `/`

- Токен вводится вручную в поле `type=password`; хранится только в памяти
  JS-переменной на время запроса. **Не** встраивается в HTML/JS, **не**
  пишется в `localStorage`, **не** передаётся в URL.
- Демонстрация на loopback допустима; при передаче токена по сети —
  только HTTPS или SSH-туннель (`ssh -L 8000:127.0.0.1:8000 mluser@vm`).

### Замаскированные значения (для отчёта)

`API_TOKEN=demo-***` (8 символов, показаны только первые 4), `X-API-Key: demo-***`.
