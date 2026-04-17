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
      pytorch-bf16-rest-nobatch.csv
      pytorch-bf16-rest-dynbatch.csv
      pytorch-bf16-grpc-dynbatch.csv
    gliner-guard-bi/
      pytorch-bf16-rest-nobatch.csv
      pytorch-bf16-rest-dynbatch.csv
      pytorch-bf16-grpc-dynbatch.csv
```

После нового прогона raw Ray Serve результаты можно разложить в этот layout
через helper-скрипт:

```bash
python3 scripts/curate_ray_results.py \
  --model gliner-guard-uni \
  --rest-nobatch ray-rest-nobatch-uni-prompts-run2 \
  --rest-dynbatch ray-rest-B4-uni-prompts-run2 \
  --grpc-dynbatch ray-grpc-B4-uni-prompts-run2
```

Аналогично для `gliner-guard-bi`.

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
- точность (fp32, fp16, int8)
- требования к железу (например: только NVIDIA GPU, минимум N GB VRAM)
- batching-стратегия (dynamic batching, max batch size)
- другие оптимизации (compilation, кастомные операторы и т.д.)
- инструкция по запуску (установка зависимостей, команда старта)
- пример запроса к API
