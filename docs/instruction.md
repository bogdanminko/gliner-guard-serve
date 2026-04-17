# Инструкция по бенчмаркингу

## Пайплайн

1. **Подготовка модели** — внести изменения в инференс (конвертация в ONNX, квантизация, смена runtime и т.д.)
2. **Обёртка в API** — оформить серверную часть через LitServe API / RayServe и тд\
Если вы работаете с рантаймом модели, попробуйте LitServe, если это невозможно, выбирете другой подходящий для нее инференс.
3. **Нагрузочное тестирование** — прогнать Locust-тест
4. **Сохранение результатов** — скачать отчёт и CSV, положить в `results/`
5. **Документация** — создать `docs/<имя-метода>.md` с описанием запуска и особенностей (см. [пример](litserve-baseline.md))
6. **Обновление README** — `make bench-readme`

## Стенд

| компонент | конфигурация |
|-----------|-------------|
| Сервер (инференс) | A100 80G SXM, 16 vCPU |
| Клиент (Locust) | отдельная виртуалка рядом, 8/16 vCPU |

Клиент запускается на отдельной машине, т.к. генерация нагрузки в 100+ RPS требует значительных ресурсов CPU.

## Нагрузочное тестирование

Тесты запускаются через [Locust](https://locust.io/):

```bash
uv run locust -f test-gliner.py -u 100 -r 1 --run-time 15m --csv=stats_name
```

| флаг | значение |
|------|----------|
| `-u 100` | 100 одновременных пользователей |
| `-r 1` | spawn rate — 1 пользователь/сек |
| `--run-time 15m` | длительность теста |
| `--csv=stats_name` | префикс для CSV-файлов со статистикой |

### Сохранение результатов

После завершения теста скачать из UI Locust (`http://localhost:8089`):

- **report.html** — полный отчёт
- **requests.csv** — статистика по запросам (используется для генерации таблицы в README)

Сохранить raw файлы в `results/` с понятным именем:

```
results/
├── litserve-baseline.csv    # requests.csv из Locust
└── litserve-baseline.html   # report.html из Locust
```

### Curated layout для README

Генератор таблицы в этом репозитории проходит по `results/**/*.csv`, поэтому
для итогового README стоит хранить только отобранные benchmark-артефакты в
отдельной структуре:

```text
results/
  ray-serve/
    gliner-guard-uni/
      pytorch-fp16-rest-nobatch.csv
      pytorch-fp16-rest-dynbatch.csv
      pytorch-fp16-grpc-dynbatch.csv
      pytorch-bf16-rest-nobatch.csv
      pytorch-bf16-rest-dynbatch.csv
      pytorch-bf16-grpc-dynbatch.csv
    gliner-guard-bi/
      pytorch-fp16-rest-nobatch.csv
      pytorch-fp16-rest-dynbatch.csv
      pytorch-fp16-grpc-dynbatch.csv
      pytorch-bf16-rest-nobatch.csv
      pytorch-bf16-rest-dynbatch.csv
      pytorch-bf16-grpc-dynbatch.csv
```

После нового прогона raw Ray Serve результаты можно разложить в этот layout
через helper-скрипт:

```bash
python3 scripts/curate_ray_results.py \
  --source-dir results \
  --model gliner-guard-uni \
  --dtype bf16 \
  --rest-nobatch ray-rest-bf16-nobatch-uni-prompts-run2 \
  --rest-dynbatch ray-rest-bf16-B4-uni-prompts-run2 \
  --grpc-dynbatch ray-grpc-bf16-B16-uni-prompts-run2
```

Для one-to-one сравнения против `fp16` baseline повторите тот же шаг с
`--dtype fp16`. Raw Ray Serve префиксы тоже стоит писать с dtype в имени,
чтобы `fp16` и `bf16` прогоны не перезаписывали друг друга.

### No-Docker fallback для Runpod

Если на GPU VM Docker недоступен или нестабилен, используйте no-Docker раннер:

```bash
REPEATS=1 USERS=100 DURATION=15m ./scripts/run-nodocker-benchmarks.sh
```

Что делает скрипт:

- клонирует ветку `feat/ray-serve-uni-bi` на локальный диск VM (`/root/...`), а не в `/workspace`
- по умолчанию тянет актуальную PR-ветку из `adapstory/gliner-guard-serve`
- поднимает Ray Serve напрямую через `uv` и локальные `.venv`
- прогоняет `uniencoder + biencoder`, `bf16 + fp16`,
  `REST nobatch + REST dynbatch + gRPC dynbatch`
- складывает raw-артефакты в `artifacts/raw-results/`
- автоматически раскладывает curated CSV/HTML в `results/ray-serve/...`
- в конце обновляет `README.md` через `make bench-readme`

Для Runpod это предпочтительнее запуска из `/workspace`, потому что там volume
смонтирован через `fuse`, а heavy Python/Ray/Torch окружения заметно медленнее
стартуют с сетевой файловой системы.

### Обновление таблицы в README

```bash
make bench-readme
```

Скрипт проходит по всем `results/**/*.csv`, извлекает строку `Aggregated` и
обновляет таблицу в README с колонками `RPS`, `P50`, `P95`, `P99`,
`Err rate`.

## Документация метода инференса

При добавлении нового бенчмарка создайте файл `docs/<имя-метода>.md` по аналогии с [docs/litserve-baseline.md](litserve-baseline.md). Что стоит указать:

- формат модели (PyTorch, ONNX, TensorRT и т.д.)
- точность (fp32, fp16, bf16, int8)
- требования к железу (например: только NVIDIA GPU, минимум N GB VRAM)
- batching-стратегия (dynamic batching, max batch size)
- другие оптимизации (compilation, кастомные операторы и т.д.)
- инструкция по запуску (установка зависимостей, команда старта)
- пример запроса к API
