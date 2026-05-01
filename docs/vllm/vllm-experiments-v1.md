# vLLM v1: scheduler baseline + multi-4x

Первая осмысленная серия ручных экспериментов для GLiNER Guard через vLLM.

Этот документ описывает пошаговый runbook для стартовой матрицы, где мы
сравниваем несколько single-instance scheduler-конфигов и один multi-instance
вариант через `vllm-factory-serve`.

---

## Что исследуем

Матрица v1 отвечает на простой вопрос:

```text
Какой базовый scheduler-конфиг даёт лучший throughput/latency
до более тонкого тюнинга?
```

Гипотезы этой волны:

1. `sched-safe` - консервативный baseline
2. `sched-balanced` - умеренная конкурентность
3. `sched-aggressive` - максимальный batching
4. `sched-short` - гипотеза про короткие тексты
5. `multi-4x` - первый агрессивный multi-instance вариант

Все конфиги ниже запускаются с одинаковой нагрузкой:

```text
dtype = auto
CUDA graphs = on
Locust users = 100
Locust spawn rate = 1
Locust run time = 15m
```

---

## Стенд

| компонент | конфигурация | роль |
|-----------|-------------|------|
| GPU pod | A100 80G SXM | поднимает `vllm serve` или `vllm-factory-serve` |
| CPU pod | CPU only | запускает `Locust` |

Связь между pod идёт через `Global Networking`.

Базовый host GPU pod:

```text
<POD_ID>.runpod.internal
```

Если pod пересоздан, подставь новый `POD_ID`.

---

## Матрица экспериментов

| Config | Имя | Instances | max-model-len | max-num-seqs | max-num-batched-tokens | Идея |
|---|---|:---:|---:|---:|---:|---|
| 1 | `sched-safe` | 1 | 8192 | 64 | 16384 | безопасный baseline |
| 2 | `sched-balanced` | 1 | 8192 | 128 | 32768 | умеренная конкурентность |
| 3 | `sched-aggressive` | 1 | 8192 | 256 | 65536 | максимум batching |
| 4 | `sched-short` | 1 | 4096 | 256 | 65536 | короткие тексты |
| 5 | `multi-4x` | 4 | 8192 | 64/inst | 32768 | multi-instance вариант |

### Что крутим

| Параметр | Что делает |
|---|---|
| `--max-model-len` | Максимальная длина входа |
| `--max-num-seqs` | Макс. число одновременных запросов в scheduler |
| `--max-num-batched-tokens` | Бюджет токенов на одну scheduler-итерацию |
| `--num-instances` | Число процессов vLLM на одном GPU |

---

## Подготовка: GPU pod

### 1. Подключение

```bash
ssh <gpu-pod-ssh-address> -i ~/.ssh/id_ed25519
```

### 2. Клон и установка

```bash
git clone -b feature/vllm_inference https://github.com/Reterno12/gliner-guard-serve.git
cd gliner-guard-serve/vllm
./setup.sh
```

### 3. Активировать среду

```bash
cd ~/gliner-guard-serve
source .venv/bin/activate
```

### 4. Подготовить модель один раз

```bash
cd ~/gliner-guard-serve/vllm
./serve.sh
```

Дождаться готовности сервера и остановить его `Ctrl+C`.

После этого должна появиться подготовленная модель:

```text
/tmp/gliner-guard-uni-vllm
```

### 5. Узнать internal DNS GPU pod

```text
<POD_ID>.runpod.internal
```

---

## Подготовка: CPU pod

### 1. Подключение

```bash
ssh <cpu-pod-ssh-address> -i ~/.ssh/id_ed25519
```

### 2. Клон и установка зависимостей

```bash
git clone -b feature/vllm_inference https://github.com/Reterno12/gliner-guard-serve.git
cd gliner-guard-serve/vllm
./setup.sh --no-vllm
```

### 3. Проверить сеть до GPU pod

```bash
getent hosts <POD_ID>.runpod.internal
curl --max-time 5 -sf http://<POD_ID>.runpod.internal:8000/health && echo "OK" || echo "FAIL"
```

Если сервер ещё не поднят, `curl` может вернуть `FAIL`. Это нормально на этапе
до pre-flight. Важнее, чтобы DNS-resolve работал.

---

## Pre-flight перед основной матрицей

Перед полным прогоном стоит проверить, что сервер вообще поднимается и
межподовая сеть работает.

### Шаг 1. Поднять тестовый сервер на GPU pod

```bash
cd ~/gliner-guard-serve/vllm
./serve.sh
```

Дождаться строки `Application startup complete`.

### Шаг 2. Проверить запрос с GPU pod

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

### Шаг 3. Проверить запрос с CPU pod

```bash
curl -s http://<POD_ID>.runpod.internal:8000/pooling \
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

### Шаг 4. Короткий Locust smoke test

```bash
cd ~/gliner-guard-serve/test-script
GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 10 \
  -r 1 \
  --run-time 1m \
  --csv=/tmp/preflight-v1
```

Проверить:

- `Failure Count = 0`
- `Requests/s > 0`
- нет таймаутов

### Шаг 5. Остановить тестовый сервер

На GPU pod: `Ctrl+C`.

---

## Общий цикл для любого конфига

### 1. На GPU pod поднять сервер с нужными флагами

Запускается ровно один конфиг за раз.

### 2. На CPU pod запустить `Locust`

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 15m \
  --csv=/tmp/<experiment-name> \
  --html=/tmp/<experiment-name>.html \
  --csv-full-history
```

### 3. После завершения Locust

На CPU pod должны появиться файлы:

- `/tmp/<experiment-name>_stats.csv`
- `/tmp/<experiment-name>_stats_history.csv`
- `/tmp/<experiment-name>_failures.csv`
- `/tmp/<experiment-name>_exceptions.csv`
- `/tmp/<experiment-name>.html`

### 4. Остановить сервер на GPU pod

Если процесс в foreground, просто `Ctrl+C`.

### 5. Проверить, что порт освободился

```bash
ss -ltnp | grep ':8000'
```

Если вывода нет, можно запускать следующий конфиг.

---

## Config 1: `sched-safe`

Идея:

- консервативный scheduler
- ниже риск memory pressure
- хорошая стартовая точка для сравнения

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm serve /tmp/gliner-guard-uni-vllm \
  --runner pooling \
  --trust-remote-code \
  --dtype auto \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.80 \
  --io-processor-plugin mmbert_gliner2_io \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384
```

### CPU pod

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
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

---

## Config 2: `sched-balanced`

Идея:

- умеренная конкурентность
- основной кандидат на production-like baseline

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm serve /tmp/gliner-guard-uni-vllm \
  --runner pooling \
  --trust-remote-code \
  --dtype auto \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.80 \
  --io-processor-plugin mmbert_gliner2_io \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768
```

### CPU pod

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 15m \
  --csv=/tmp/sched-balanced \
  --html=/tmp/sched-balanced.html \
  --csv-full-history
```

---

## Config 3: `sched-aggressive`

Идея:

- максимальный batching
- наибольший риск по latency и memory pressure

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm serve /tmp/gliner-guard-uni-vllm \
  --runner pooling \
  --trust-remote-code \
  --dtype auto \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.80 \
  --io-processor-plugin mmbert_gliner2_io \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 65536
```

### CPU pod

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 15m \
  --csv=/tmp/sched-aggressive \
  --html=/tmp/sched-aggressive.html \
  --csv-full-history
```

---

## Config 4: `sched-short`

Идея:

- предполагаем, что реальные тексты короче
- уменьшаем `max-model-len`
- проверяем, даст ли это лучшую плотность serving

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm serve /tmp/gliner-guard-uni-vllm \
  --runner pooling \
  --trust-remote-code \
  --dtype auto \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.80 \
  --io-processor-plugin mmbert_gliner2_io \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 65536
```

### CPU pod

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 15m \
  --csv=/tmp/sched-short \
  --html=/tmp/sched-short.html \
  --csv-full-history
```

---

## Config 5: `multi-4x`

Идея:

- 4 instance на одном GPU
- reverse proxy через `vllm-factory-serve`
- проверка, выигрывает ли encoder workload от multi-instance

Важно:

- это не `vllm serve`
- это `vllm-factory-serve`
- предыдущий single-instance сервер должен быть полностью остановлен

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-batched-tokens 32768 \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling --no-enable-prefix-caching --no-enable-chunked-prefill
```

### CPU pod

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 15m \
  --csv=/tmp/multi-4x \
  --html=/tmp/multi-4x.html \
  --csv-full-history
```

---

## Рекомендуемый порядок запуска

1. `sched-safe`
2. `sched-balanced`
3. `sched-aggressive`
4. `sched-short`
5. `multi-4x`

Такой порядок идёт от самого безопасного single-instance baseline к наиболее
агрессивной конфигурации.

---

## Сводка по фактическим результатам

<!-- AUTO_VLLM_RESULTS:START -->

Источник: `results/vllm/gliner-guard_v1`

| Experiment | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Err rate (%) |
|---|---:|---:|---:|---:|---:|
| bfloat16-eager | 107.6 | 870 | 1100 | **1100** | 0.00 |
| float16-eager | 105.5 | 900 | 1100 | **1100** | 0.00 |
| bfloat16-cudagraph | 106.8 | 880 | 1100 | **1100** | 0.00 |
| float16-cudagraph | **108.6** | **860** | 1100 | **1100** | 0.00 |
| bfloat16-eager-batch16k | 88.6 | 1100 | 1200 | 1200 | 0.00 |
| bfloat16-eager-mem90 | 108.3 | 870 | **1000** | **1100** | 0.00 |

Короткий вывод:
- лучший throughput: `float16-cudagraph`
- лучший p95: `bfloat16-eager-mem90`
- конфиги с failures: нет

<!-- AUTO_VLLM_RESULTS:END -->

---

## Что сохранять после каждого прогона

Минимум:

- `*_stats.csv`
- `*_stats_history.csv`
- `*.html`

Полный набор:

- `*_failures.csv`
- `*_exceptions.csv`
- лог сервера, если запуск шёл не в screen/tmux

---

## Как принимать решение

### Шаг 1. Сначала убрать явно плохие конфиги

Сразу отбрасывай всё, где:

- есть failures
- есть частые таймауты
- p95/p99 слишком резко хуже baseline

### Шаг 2. Сравнить throughput и latency

Смотреть минимум на:

- `Requests/s`
- `Average Response Time`
- `95%`
- `99%`

### Шаг 3. Отдельно оценить `multi-4x`

Если `multi-4x` даёт выше throughput без развала p95/p99, это сильный кандидат.
Если tail latency сильно хуже, лучше оставить single-instance baseline для
следующей волны.

---

## Troubleshooting

- Если новый конфиг не стартует, проверь, что предыдущий сервер действительно
  остановлен и порт `8000` свободен.
- Если `Locust` не видит GPU pod, проверь `Global Networking` и актуальный
  `POD_ID.runpod.internal`.
- Если `vllm-factory-serve` падает, сначала убедись, что single-instance
  `vllm serve` вообще поднимается на том же подготовленном `/tmp`-артефакте.

---

## Короткий итог

`v1` - это стартовая матрица для грубого отбора. Она нужна не для идеального
production-конфига, а чтобы быстро понять:

1. насколько чувствителен стенд к scheduler limits
2. помогает ли меньший `max-model-len`
3. есть ли смысл дальше развивать multi-instance путь
