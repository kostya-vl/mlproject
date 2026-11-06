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
