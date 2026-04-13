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
- Установит vllm-factory с GLiNER-зависимостями
- Установит Locust и зависимости тестов
- Установит vLLM (последним, чтобы зафиксировать совместимые версии)

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

Скрипт `vllm/experiments.sh` запускает серию экспериментов с разными конфигурациями:

| Эксперимент | Описание |
|---|---|
| `bfloat16-eager` | Базовый: bfloat16 + eager mode (без CUDA graphs) |
| `float16-eager` | float16 + eager mode |
| `bfloat16-cudagraph` | bfloat16 + CUDA graphs (потенциал для малых батчей) |
| `float16-cudagraph` | float16 + CUDA graphs |
| `bfloat16-eager-batch16k` | bfloat16 + увеличенный бюджет токенов (16384) |
| `bfloat16-eager-batch4k` | bfloat16 + уменьшенный бюджет токенов (4096, меньше латенси) |
| `bfloat16-eager-mem90` | bfloat16 + 90% GPU memory |

### Запуск всех экспериментов

```bash
cd vllm
./experiments.sh
```

### Запуск одного эксперимента

```bash
./experiments.sh bfloat16-eager
```

### Результаты

Результаты сохраняются в `results/vllm/gliner-guard-uni/`:

```
results/vllm/gliner-guard-uni/
├── bfloat16-eager_stats.csv        # Locust stats
├── bfloat16-eager-server.log       # vLLM server log
├── bfloat16-eager-locust.log       # Locust output
├── float16-eager_stats.csv
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

```bash
# Все 7 конфигураций (bfloat16-eager, float16-eager, CUDA graphs, и т.д.)
./experiments.sh

# Или один конкретный эксперимент
./experiments.sh bfloat16-eager
```

Каждый эксперимент:
1. Запускает vLLM сервер с заданными параметрами
2. Ждёт health-check + warmup (30 сек)
3. Запускает Locust (100 пользователей, 15 мин)
4. Сохраняет результаты и убивает сервер

### 5. Результаты

```
results/vllm/gliner-guard-uni/
├── bfloat16-eager_stats.csv         # Сводная статистика Locust
├── bfloat16-eager_stats_history.csv # Посекундная история
├── bfloat16-eager-server.log        # Логи vLLM
├── bfloat16-eager-locust.log        # Логи Locust
├── float16-eager_stats.csv
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
