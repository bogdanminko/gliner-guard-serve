# vLLM v2: scheduler tuning + multi-instance

Вторая серия экспериментов: тюнинг scheduler-параметров vLLM и multi-instance serving через `vllm-factory-serve`.

## Стенд

| компонент | конфигурация | RunPod |
|-----------|-------------|--------|
| GPU pod (сервер) | A100 80G SXM, 16 vCPU | GPU pod |
| CPU pod (Locust) | 8 vCPU, без GPU | CPU pod, та же сеть |

Locust вынесен на отдельный CPU pod, чтобы генерация нагрузки не конкурировала с инференсом за CPU.

## Матрица экспериментов

| Эксперимент | Instances | max-model-len | max-num-seqs | max-num-batched-tokens | Идея |
|---|:---:|---:|---:|---:|---|
| `sched-safe` | 1 | 8192 | 64 | 16384 | Консервативные лимиты, низкий memory pressure |
| `sched-balanced` | 1 | 8192 | 128 | 32768 | Умеренная конкурентность |
| `sched-aggressive` | 1 | 8192 | 256 | 65536 | Максимальный batching |
| `sched-short` | 1 | 4096 | 256 | 65536 | Короткие тексты: меньше model-len → больше seqs в память |
| `multi-4x` | 4 | 8192 | 64/inst | 32768 | 4 vLLM instance за reverse proxy |

Все эксперименты: `float16` + CUDA graphs (`--enforce-eager` не используем; это baseline).

### Что крутим

| Параметр | Что делает |
|---|---|
| `--max-model-len` | Максимальная длина входа. Меньше → больше seqs в один батч |
| `--max-num-seqs` | Макс. число одновременных запросов в scheduler |
| `--max-num-batched-tokens` | Бюджет токенов на одну итерацию scheduler. Больше → толще батч |
| `--num-instances` | Число vLLM process на одном GPU (через `vllm-factory-serve`) |

## Подготовка: GPU pod

### 1. Подключение

```bash
ssh <gpu-pod-ssh-address> -i ~/.ssh/id_ed25519
```

### 2. Клон и установка (один раз)

```bash
git clone -b feature/vllm_inference https://github.com/Reterno12/gliner-guard-serve.git
cd gliner-guard-serve/vllm
./setup.sh
```
### 2.5  Активировать среду
cd gliner-guard-serve/
source .venv/bin/activate

### 3. Подготовка модели (один раз)

```bash
cd gliner-guard-serve/vllm
./serve.sh
# дождаться "Application startup complete", затем Ctrl+C
```

Модель сохранится в `/tmp/gliner-guard-uni-vllm`.

### 4. Узнать internal DNS GPU pod

Использовать hostname вида:

```text
<GPU_POD_ID>.runpod.internal
```

Это основной адрес для CPU pod при включённом `Global Networking`.

Если нужен fallback для старого стенда без internal DNS, можно взять IP:

```bash
hostname -I | awk '{print $1}'
```

## Подготовка: CPU pod

### 1. Создать CPU pod в RunPod

- Template: любой с Python 3.8+ для `Locust`, Python 3.11+ не обязателен
- GPU: не нужен (CPU only, 8 vCPU достаточно)
- Включить `Global Networking` при создании pod
- Убедиться, что оба pod подняты в совместимом окружении RunPod

### 2. Подключение

```bash
ssh <cpu-pod-ssh-address> -i ~/.ssh/id_ed25519
```

### 3. Клон и установка зависимостей

```bash
git clone -b feature/vllm_inference https://github.com/Reterno12/gliner-guard-serve.git
cd gliner-guard-serve/vllm
./setup.sh --no-vllm
```

Это ставит только зависимости для `Locust` и не тянет `vllm` / `vllm-factory`, поэтому подходит для CPU-only pod.

### 4. Проверить сетевую связность с GPU pod

```bash
getent hosts <GPU_POD_ID>.runpod.internal
curl --max-time 5 -sf http://<GPU_POD_ID>.runpod.internal:8000/health && echo "OK" || echo "FAIL"
```

Если `FAIL`:
- проверить, что сервер запущен на GPU pod
- проверить, что сервис слушает `0.0.0.0:8000`
- проверить, что `Global Networking` включён на обоих pod

## Pre-flight: проверка перед основным прогоном

Все действия ниже — **до** запуска полной матрицы. Цель: убедиться, что оба пода настроены, сервер стартует, Locust доходит до GPU pod.

### Шаг 1. Запустить сервер на GPU pod

```bash
cd gliner-guard-serve/vllm
./serve.sh
```

Дождаться `Application startup complete`.

### Шаг 2. Проверить одиночный запрос (с GPU pod)

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

Ожидание: JSON с entities и classification. Если ошибка — смотреть лог сервера.

### Шаг 3. Проверить запрос с CPU pod

```bash
curl -s http://<GPU_POD_ID>.runpod.internal:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/gliner-guard-uni-vllm",
    "task": "plugin",
    "data": {
      "text": "Call me at +1-555-0100",
      "schema": {
        "entities": ["person", "address", "email", "phone"],
        "classifications": [{"task": "safety", "labels": ["safe", "unsafe"]}]
      },
      "threshold": 0.4
    }
  }' | python3 -m json.tool
```

Если ответ пришёл — сеть между подами работает.

### Шаг 4. Короткий Locust-прогон с CPU pod (1 минута)

```bash
cd gliner-guard-serve/test-script
GLINER_HOST=http://<GPU_POD_ID>.runpod.internal:8000 \
python -m locust \
    -f test-gliner-vllm.py \
    --headless \
    -u 10 \
    -r 1 \
    --run-time 1m \
    --csv=/tmp/preflight
```

Проверить:
- `Failure Count` = 0
- `Requests/s` > 0
- Нет таймаутов

Если всё ОК — готовы к полному прогону.

### Шаг 5. Остановить тестовый сервер

На GPU pod: `Ctrl+C` в терминале с `serve.sh`.

## Запуск экспериментов

### Вариант A: автоматический (с SSH между подами, трафик к vLLM через Global Networking)

Настроить SSH-ключ GPU pod → CPU pod:

```bash
# На GPU pod
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519  # если ещё нет
ssh-copy-id -p <CPU_SSH_PORT> root@<CPU_PUBLIC_IP>
ssh -p <CPU_SSH_PORT> root@<CPU_PUBLIC_IP> echo "SSH OK"
```

Запуск:

```bash
cd gliner-guard-serve/vllm

LOCUST_SSH=root@<CPU_PUBLIC_IP> \
LOCUST_SSH_PORT=<CPU_SSH_PORT> \
GPU_POD_HOST=<GPU_POD_ID>.runpod.internal \
REMOTE_TEST_DIR=~/gliner-guard-serve/test-script \
./experiments.sh
```

Скрипт для каждого эксперимента:
1. Поднимает vLLM сервер (или vllm-factory-serve для multi-instance)
2. Ждёт healthcheck + warmup 30 сек
3. Запускает Locust на CPU pod через SSH
4. Копирует результаты обратно на GPU pod через scp
5. Останавливает сервер
6. Переходит к следующему эксперименту

Один конкретный эксперимент:

```bash
LOCUST_SSH=root@<CPU_PUBLIC_IP> \
LOCUST_SSH_PORT=<CPU_SSH_PORT> \
GPU_POD_HOST=<GPU_POD_ID>.runpod.internal \
./experiments.sh sched-balanced
```

Список доступных экспериментов:

```bash
./experiments.sh --list
```

### Вариант B: ручной (два терминала)

Если SSH между подами не настроен или хочется контролировать каждый шаг вручную.

**Терминал 1 — GPU pod (сервер):**

```bash
cd gliner-guard-serve/vllm

# sched-safe
vllm serve /tmp/gliner-guard-uni-vllm \
    --runner pooling \
    --trust-remote-code \
    --dtype float16 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.80 \
    --io-processor-plugin mmbert_gliner2_io \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384
```

**Терминал 2 — CPU pod (Locust):**

```bash
cd gliner-guard-serve/test-script

GLINER_HOST=http://<GPU_POD_ID>.runpod.internal:8000 \
python -m locust \
    -f test-gliner-vllm.py \
    --headless \
    -u 100 \
    -r 1 \
    --run-time 15m \
    --csv=/tmp/sched-safe \
    --html=/tmp/sched-safe.html \
    --csv-full-history
```

После завершения — `Ctrl+C` на сервере, скопировать результаты, запустить следующий конфиг.

**Для multi-4x (multi-instance):**

```bash
# GPU pod
vllm-factory-serve /tmp/gliner-guard-uni-vllm \
    --num-instances 4 \
    --max-batch-size 64 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 32768 \
    --io-processor-plugin mmbert_gliner2_io \
    -- --runner pooling --no-enable-prefix-caching --no-enable-chunked-prefill
```

Locust-команда та же, что и для single-instance.

## Результаты

```
results/vllm/gliner-guard-uni/
├── sched-safe_stats.csv
├── sched-safe_stats_history.csv
├── sched-safe-server.log
├── sched-safe-locust.log
├── sched-balanced_stats.csv
├── sched-aggressive_stats.csv
├── sched-short_stats.csv
├── multi-4x_stats.csv
└── ...
```

### Автосводка по фактическим результатам

<!-- AUTO_VLLM_RESULTS:START -->

Источник: `results/vllm/gliner-guard_v2`

| Experiment | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Err rate (%) |
|---|---:|---:|---:|---:|---:|
| sched-safe | 89.5 | 1000 | 1400 | 1500 | 0.00 |
| sched-balanced | 61.9 | 1700 | 1800 | 1900 | 0.00 |
| sched-aggressive | 60.9 | 1700 | 1800 | 2000 | 0.00 |
| sched-short | 62.8 | 1600 | 1800 | 1900 | 0.00 |
| multi-4x | **124.9** | **800** | **880** | **930** | 0.00 |

Короткий вывод:
- лучший throughput: `multi-4x`
- лучший p95: `multi-4x`
- конфиги с failures: нет

<!-- AUTO_VLLM_RESULTS:END -->

### Скачать на локальную машину

```bash
scp -r <gpu-pod-ssh>:~/gliner-guard-serve/results/ ./results/
```

## Troubleshooting

| Проблема | Что делать |
|---|---|
| `Server failed to start` | `tail -50 results/vllm/gliner-guard-uni/<name>-server.log` |
| OOM при aggressive | Уменьшить `--max-num-batched-tokens` или `--max-num-seqs` |
| multi-4x не стартует | Проверить `vllm-factory-serve --help`, возможно нужно обновить vllm-factory |
| `getent hosts <GPU_POD_ID>.runpod.internal` пустой | `Global Networking` не включён или pod пересоздан без него |
| Locust 0 req/s | Проверить `curl http://<GPU_POD_ID>.runpod.internal:8000/health` с CPU pod |
| `Connection refused` с CPU pod | Сервер ещё не стартовал или сервис слушает не `0.0.0.0` |
| `SKIP: results already exist` | Удалить `results/vllm/gliner-guard-uni/<name>_stats.csv` |
