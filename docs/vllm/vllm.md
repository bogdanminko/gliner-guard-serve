# vLLM + vllm-factory

Инференс GLiNER Guard через [vLLM](https://github.com/vllm-project/vllm) с использованием плагина `mmbert_gliner2` из [vllm-factory](https://github.com/ddickmann/vllm-factory).

## Зачем vLLM для BERT-like моделей

vLLM изначально создан для авторегрессионных LLM, но через pooling runner и vllm-factory поддерживает encoder-модели (BERT, DeBERTa, ModernBERT). Ключевые оптимизации для encoder-моделей:

| Оптимизация | Описание | Релевантность для BERT |
|---|---|---|
| Continuous batching | Динамическое формирование батчей без ожидания заполнения | Высокая — основное преимущество |
| CUDA graphs | Захват и повторное воспроизведение GPU-операций | Средняя — ускоряет при малых батчах |
| Tensor parallelism | Разделение модели на несколько GPU | Низкая (модель маленькая) |
| Prefix caching | Кэширование KV-cache для общих префиксов | Не применимо (encoder, нет KV-cache) |
| Quantization | INT8/INT4 квантизация весов | Перспективная |
| Multi-instance | Несколько процессов vLLM на одном GPU | Высокая для memory-bound моделей |

## Особенности

- **Модель:** `hivetrace/gliner-guard-uniencoder` (GLiNER2 + ModernBERT backbone)
- **Плагин:** `mmbert_gliner2` (vllm-factory) — ModernBERT backbone + GLiNER2 pooler
- **Формат:** PyTorch (safetensors), конвертация через `vllm-factory-prep`
- **Точность:** bfloat16 / float16
- **Требования:** NVIDIA GPU, CUDA 12+, vLLM >= 0.19
- **Batching:** continuous batching (vLLM scheduler)

### Схема модели

- **PII-сущности:** person, address, email, phone (threshold 0.4)
- **Классификация:** safety (safe / unsafe)
- **Формат запроса:** GLiNER2 schema (entities + classifications)

## Установка

### На RunPod / GPU VM

**Требования:** NVIDIA GPU (A100 80GB рекомендуется), CUDA 12+, Python 3.11+

```bash
# 1. Клонировать репозиторий (ветка feature/vllm_inference)
git clone -b feature/vllm_inference https://github.com/Reterno12/gliner-guard-serve.git
cd gliner-guard-serve

# 2. Создать venv и установить зависимости
python3 -m venv .venv
source .venv/bin/activate

# 3. Установить всё через setup.sh
cd vllm
./setup.sh
```

Скрипт `setup.sh` автоматически:
- Установит `vllm-factory` с GLiNER-зависимостями
- Установит `Locust` и зависимости тестов
- Установит протестированную версию `vLLM` (`0.19.1`) вместе с совместимой для `GLiNER` версией `transformers` (`>=4.56,<5.0`)
- Выполнит `pip check`, чтобы сразу поймать конфликт зависимостей

> **Примечание:** На RunPod с PyTorch pre-installed можно пропустить создание venv и устанавливать прямо в системный Python.
> Если `python3` не 3.11+, проверьте `python3.11 --version` и используйте его для venv.

## Запуск сервера

### Быстрый старт

```bash
cd vllm
./serve.sh
```

Скрипт автоматически подготовит модель (при первом запуске) и запустит vLLM.

### Ручной запуск

```bash
# Подготовка модели (один раз)
vllm-factory-prep \
    --model hivetrace/gliner-guard-uniencoder \
    --plugin mmbert_gliner2 \
    --output /tmp/gliner-guard-uni-vllm

# Запуск сервера
vllm serve /tmp/gliner-guard-uni-vllm \
    --runner pooling \
    --trust-remote-code \
    --dtype bfloat16 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.80 \
    --io-processor-plugin mmbert_gliner2_io
```

Сервер стартует на `http://localhost:8000`.

### Пример запроса

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/gliner-guard-uni-vllm",
    "task": "plugin",
    "data": {
      "text": "Send $500 to John Smith at john.smith@gmail.com",
      "schema": {
        "entities": ["person", "address", "email", "phone"],
        "classifications": [{"task": "safety", "labels": ["safe", "unsafe"]}]
      },
      "threshold": 0.4
    }
  }'
```

## Нагрузочное тестирование

```bash
cd test-script
GLINER_HOST=http://localhost:8000 \
uv run locust -f test-gliner-vllm.py -u 100 -r 1 --run-time 15m --csv=vllm-stats
```

Тест использует те же данные (`prompts.csv`, `responses.csv`), что и для LitServe/ONNX.

## Матрица экспериментов

Ниже не "автоматические пресеты", а разные ручные серии прогонов, которые
оформлены в отдельные runbook-файлы.

### v1: стартовая ручная baseline-матрица

| Эксперимент | Описание |
|---|---|
| `bfloat16-eager` | Базовый: bfloat16 + eager mode (без CUDA graphs) |
| `float16-eager` | float16 + eager mode |
| `bfloat16-cudagraph` | bfloat16 + CUDA graphs |
| `float16-cudagraph` | float16 + CUDA graphs |
| `bfloat16-eager-batch16k` | bfloat16 + увеличенный token budget (16384) |
| `bfloat16-eager-mem90` | bfloat16 + 90% GPU memory |

Файл: `vllm-experiments-v1.md`

Смысл серии: сначала понять, как ведёт себя самый ранний baseline на разных
режимах dtype / eager / cudagraph и грубых лимитах.

### v2: ручной прогон scheduler tuning + multi-instance

| Эксперимент | Instances | max-model-len | max-num-seqs | max-num-batched-tokens |
|---|:---:|---:|---:|---:|
| `sched-safe` | 1 | 8192 | 64 | 16384 |
| `sched-balanced` | 1 | 8192 | 128 | 32768 |
| `sched-aggressive` | 1 | 8192 | 256 | 65536 |
| `sched-short` | 1 | 4096 | 256 | 65536 |
| `multi-4x` | 4 | 8192 | 64/inst | 32768 |

Файл: `vllm-experiments-v2.md`

Смысл серии: прогнать конфиги руками на двухподовом стенде и сравнить
single-instance против multi-instance.

### v3: сокращённая ручная осмысленная матрица

| Эксперимент | Идея |
|---|---|
| `single-safe` | стабильный single-instance baseline |
| `single-dense` | плотный single-instance режим |
| `single-short` | гипотеза для коротких текстов |
| `multi-2x` | мягкий multi-instance |
| `multi-4x` | агрессивный multi-instance |

Файл: `vllm-experiments-v3.md`

Смысл серии: оставить только конфиги, которые отличаются по стратегии serving,
а не просто слегка двигают близкие цифры.

### v4: финальный ручной runbook для fixed-batch тюнинга

| Волна | Что сравниваем |
|---|---|
| Config 0-5 | длину контекста и token budget |
| Config 6-7 | A/B-диагностику лучшего конфига |

Файл: `vllm_experiments-v4-final.md`

Смысл серии: вручную выбрать лучший practical config для GLiNER2/mBERT pooling
нагрузки и потом добить его диагностическими A/B тестами.

### Как запускать на практике

На практике здесь используется ручной цикл:

1. Открыть нужный runbook (`v1`, `v2`, `v3` или `v4`)
2. На GPU pod поднять сервер для одного конкретного конфига
3. На CPU pod запустить `Locust` для этого же конфига
4. Дождаться завершения, сохранить артефакты
5. Остановить сервер
6. Перейти к следующему конфигу

То есть запуск идёт не как "одна кнопка прогнать всё", а как ручная серия
контролируемых прогонов.

Если нужен именно автоматический проход, можно использовать `./experiments.sh`,
но это вспомогательный путь, а не основной workflow этого проекта.

### Результаты

Результаты сохраняются в `results/vllm/gliner-guard-uni/`:

```
results/vllm/gliner-guard-uni/
├── sched-safe_stats.csv            # Locust stats
├── sched-safe_stats_history.csv    # Per-second history
├── sched-safe-server.log           # vLLM server log
├── sched-safe-locust.log           # Locust output
├── sched-balanced_stats.csv
├── multi-4x_stats.csv
└── ...
```

## Архитектура

```
                    ┌─────────────────────────────────────────┐
                    │          vLLM (pooling runner)           │
  POST /pooling     │                                         │
  ───────────────►  │  IOProcessor (mmbert_gliner2_io)         │
  {model, task,     │    ├─ factory_parse: schema → labels    │
   data: {text,     │    ├─ factory_pre_process: tokenize     │
   schema, ...}}    │    │                                    │
                    │  Engine                                  │
                    │    ├─ Scheduler (continuous batching)    │
                    │    ├─ ModernBERT encoder (backbone)      │
                    │    └─ GLiNER2 pooler (span extraction)   │
                    │                                         │
  ◄───────────────  │  IOProcessor.factory_post_process       │
  [{entity, ...}]   │    └─ decode spans → entities + classes  │
                    └─────────────────────────────────────────┘
```

## Сравнение с LitServe

| Аспект | LitServe | vLLM |
|---|---|---|
| Batching | Dynamic (max_batch_size=64, timeout=50ms) | Continuous (scheduler-driven) |
| Workers | 4 per device | Один процесс, async |
| Overhead | Python GIL, worker coordination | C++ scheduler, zero-copy |
| Гибкость | Произвольный код в predict() | Фиксированный pipeline |
| Формат API | POST /predict {text} | POST /pooling {model, data, task} |

## Запуск на RunPod: пошаговый гайд

### 1. Подключение

```bash
ssh <runpod-ssh-address> -i ~/.ssh/id_ed25519
```

> Порт может меняться при перезапуске пода. Актуальный адрес — в RunPod UI → Connect → SSH Terminal.

### 2. Установка (один раз)

```bash
git clone -b feature/vllm_inference https://github.com/Reterno12/gliner-guard-serve.git
cd gliner-guard-serve/vllm
./setup.sh
```

Для отдельного CPU pod с одним `Locust` используйте:

```bash
cd gliner-guard-serve/vllm
./setup.sh --no-vllm
```

### 3. Проверка: одиночный запрос

```bash
# Запустить сервер (подготовит модель при первом запуске)
./serve.sh
```

Дождаться лога `Application startup complete`, затем в другом терминале:

```bash
curl -s http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/gliner-guard-uni-vllm",
    "task": "plugin",
    "data": {
      "text": "Send $500 to John Smith at john.smith@gmail.com",
      "schema": {
        "entities": ["person", "address", "email", "phone"],
        "classifications": [{"task": "safety", "labels": ["safe", "unsafe"]}]
      },
      "threshold": 0.4
    }
  }' | python3 -m json.tool
```

Ожидаемый ответ: JSON с найденными entities и classification.

### 4. Полный прогон экспериментов

Дальше обычно работаешь не через один общий скрипт, а через конкретный runbook.

Практический шаблон такой:

1. Выбрать нужный markdown из этой папки
2. На GPU pod запустить команду сервера для одного конфига
3. На CPU pod запустить `Locust` для этого конфига
4. Дождаться конца прогона
5. Сохранить `stats`, `stats_history`, HTML и при необходимости GPU trace
6. Остановить сервер и перейти к следующему конфигу

Каждый эксперимент:
1. Запускает vLLM сервер с заданными параметрами
2. Проходит health-check / одиночный smoke request
3. Запускает `Locust` (обычно 100 пользователей, 15 минут)
4. Сохраняет результаты
5. После завершения сервер останавливается вручную

Подробные инструкции по ручному двухподовому стенду:

- `vllm-experiments-v2.md`
- `vllm-experiments-v3.md`
- `vllm_experiments-v4-final.md`

### 5. Результаты

```
results/vllm/gliner-guard-uni/
├── sched-safe_stats.csv             # Сводная статистика Locust
├── sched-safe_stats_history.csv     # Посекундная история
├── sched-safe-server.log            # Логи vLLM
├── sched-safe-locust.log            # Логи Locust
├── sched-balanced_stats.csv
└── ...
```

### 6. Скачать результаты

```bash
# С локальной машины
scp -r <runpod-ssh-address>:~/gliner-guard-serve/results/ ./results/
```

## Примечания

- **Prefix caching отключён** — encoder-модели не используют KV-cache, кэширование префиксов не даёт выигрыша.
- **Chunked prefill отключён** — для encoder-моделей весь вход обрабатывается целиком.
- **CUDA graphs** (`--enforce-eager` убран) — могут дать выигрыш при фиксированных input shapes, но GLiNER работает с переменной длиной входа, эффект нужно измерить.
- **Модель готовится один раз** — `vllm-factory-prep` перезаписывает `config.json` для совместимости с vLLM. Результат кэшируется в `/tmp/gliner-guard-uni-vllm`.
- **Плагин `mmbert_gliner2`** — кастомный плагин для ModernBERT + GLiNER2. Включён в `vllm-factory/` в этом репозитории (upstream vllm-factory его не содержит).
