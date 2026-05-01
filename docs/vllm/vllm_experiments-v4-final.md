# vLLM manual runbook: GLiNER2/mBERT pooling fixed-batch matrix

Этот файл описывает **пошаговый ручной запуск** серии экспериментов для
**GLiNER2 / ModernBERT / pooling** workload через `vllm-factory-serve`.

Документ написан в формате runbook: любой человек должен суметь поднять стенд,
проверить его, последовательно прогнать конфиги и понять, как выбрать следующий шаг.

---

## Что исследуем

Главное ограничение этой матрицы:

- `Locust` не меняем
- `--max-batch-size 64` не меняем
- `--num-instances 4` не меняем

Поэтому эта матрица **не про batch-size descent**.  
Она про:

1. `max-model-len`
2. `max-num-batched-tokens`
3. chunked prefill A/B
4. CUDA graphs vs eager A/B

---

## Стенд

| компонент | конфигурация | роль |
|-----------|-------------|------|
| GPU pod | A100 80G SXM | поднимает `vllm-factory-serve` |
| CPU pod | CPU only | запускает `Locust` |

Связь между pod идёт через `Global Networking`.

Базовый host GPU pod:

```text
<POD_ID>.runpod.internal
```

Если GPU pod пересоздан, подставь новый `POD_ID`.

---

## Матрица экспериментов

| Config | Имя | max-model-len | max-num-batched-tokens | Идея |
|---|---|---:|---:|---|
| 0 | `baseline-len8192-tokens262k` | 8192 | 262144 | текущий reference |
| 1 | `len4096-tokens131k` | 4096 | 131072 | уменьшить len и token budget |
| 2 | `len2048-tokens131k` | 2048 | 131072 | ещё более короткий len |
| 3 | `len4096-tokens65k` | 4096 | 65536 | умеренный len + меньше token budget |
| 4 | `len2048-tokens65k` | 2048 | 65536 | компактный production-like кандидат |
| 5 | `len2048-tokens49k` | 2048 | 49152 | latency-oriented fallback |
| 6 | `best-no-chunked-prefill` | best | best | A/B для `--no-enable-chunked-prefill` |
| 7 | `best-enforce-eager` | best | best | A/B для `--enforce-eager` |

Во всех основных конфигах фиксируем:

```text
num-instances = 4
max-batch-size = 64
dtype = float16
runner = pooling
Locust users = 100
Locust spawn rate = 1
Locust run time = 7m
```

Теоретическая batch capacity:

```text
4 instances × 64 = 256 request slots
```

Но `Locust` даёт только:

```text
-u 100
```

Поэтому request slot capacity здесь не главная ось. Главные оси:

```text
max-model-len
max-num-batched-tokens
chunked prefill
CUDA graphs / eager
```

---

## Почему это не `dense-192 / dense-160 / dense-128`

Названия вида:

```text
dense-192
dense-160
dense-128
```

имеют смысл только если меняется общий batch size.

Но здесь:

```text
max-batch-size = 64
num-instances = 4
```

всегда дают:

```text
total batch slots = 256
```

Значит, если мы меняем только `--max-num-batched-tokens`, правильнее называть
конфиги так:

```text
len4096-tokens131k
len2048-tokens65k
len2048-tokens49k
```

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

Дождаться готовности сервера, затем остановить `Ctrl+C`.

После этого модель должна лежать здесь:

```text
/tmp/gliner-guard-uni-vllm
```

### 5. Узнать internal DNS GPU pod

Использовать hostname вида:

```text
<POD_ID>.runpod.internal
```

Это основной адрес для CPU pod.

Fallback для старого стенда без internal DNS:

```bash
hostname -I | awk '{print $1}'
```

---

## Подготовка: CPU pod

### 1. Создать CPU pod

- GPU не нужен
- `Global Networking` должен быть включён
- 8 vCPU обычно достаточно для `Locust`

### 2. Подключение

```bash
ssh <cpu-pod-ssh-address> -i ~/.ssh/id_ed25519
```

### 3. Клон и установка зависимостей

```bash
git clone -b feature/vllm_inference https://github.com/Reterno12/gliner-guard-serve.git
cd gliner-guard-serve/vllm
./setup.sh --no-vllm
cd ..
source .venv/bin/activate
```

Это ставит только зависимости для `Locust`, без `vllm` и `vllm-factory`.

### 4. Проверить сеть до GPU pod

```bash
getent hosts <POD_ID>.runpod.internal
curl --max-time 5 -sf http://<POD_ID>.runpod.internal:8000/health && echo "OK" || echo "FAIL"
```

Если `FAIL`:

- проверить, что сервер реально поднят на GPU pod
- проверить, что сервис слушает `0.0.0.0:8000`
- проверить, что `Global Networking` включён на обоих pod

---

## Pre-flight перед полной матрицей

Все шаги ниже делаем **до** основного прогона. Цель: убедиться, что стенд рабочий.

### Шаг 1. Поднять тестовый сервер на GPU pod

```bash
cd ~/gliner-guard-serve/vllm
./serve.sh
```

Или, если хочешь сразу проверить multi-instance path:

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-batched-tokens 131072 \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling
```

### Шаг 2. Проверить одиночный запрос с GPU pod

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

Ожидание: приходит JSON с entities и classification.

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

Если ответ пришёл, сеть между pod работает.

### Шаг 4. Короткий Locust smoke test с CPU pod

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 10 \
  -r 1 \
  --run-time 1m \
  --csv=/tmp/preflight
```

Проверить:

- `Failure Count = 0`
- `Requests/s > 0`
- нет таймаутов и exceptions

### Шаг 5. Остановить тестовый сервер

На GPU pod:

```bash
Ctrl+C
```

---

## Общий цикл для любого конфига

Для каждого запуска делай один и тот же цикл.

### 1. На GPU pod поднять сервер с нужными флагами

Все команды ниже запускаются из:

```bash
cd ~/gliner-guard-serve/vllm
```

### 2. На CPU pod запустить `Locust`

Шаблон:

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 7m \
  --csv=/tmp/<experiment-name> \
  --html=/tmp/<experiment-name>.html \
  --csv-full-history
```

### 3. После завершения Locust

На CPU pod должны появиться файлы:

```text
/tmp/<experiment-name>_stats.csv
/tmp/<experiment-name>_stats_history.csv
/tmp/<experiment-name>_failures.csv
/tmp/<experiment-name>_exceptions.csv
/tmp/<experiment-name>.html
```

### 4. Остановить сервер на GPU pod

```bash
Ctrl+C
```

### 5. Проверить, что порт освободился

```bash
ss -ltnp | grep ':8000'
```

Если вывода нет, можно запускать следующий конфиг.

### 6. Опционально: писать GPU trace

Если хочешь сохранить `GPU util`, `VRAM`, `power`, используй обёртки из:

```text
docs/vllm_gliner2_mbert_gpu_tracking_wrappers.md
```

---

## Основная волна: Config 0-5

### Config 0: `baseline-len8192-tokens262k`

Идея:

- сохранить текущую верхнюю точку
- использовать её как baseline
- проверить, не лучше ли более компактные значения

**GPU pod**

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 262144 \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling
```

**CPU pod**

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 7m \
  --csv=/tmp/baseline-len8192-tokens262k \
  --html=/tmp/baseline-len8192-tokens262k.html \
  --csv-full-history
```

### Config 1: `len4096-tokens131k`

Идея:

- уменьшить `max-model-len` с `8192` до `4096`
- уменьшить token budget с `262k` до `131k`
- проверить, не был ли baseline слишком широким

**GPU pod**

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 131072 \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling
```

**CPU pod**

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 7m \
  --csv=/tmp/len4096-tokens131k \
  --html=/tmp/len4096-tokens131k.html \
  --csv-full-history
```

### Config 2: `len2048-tokens131k`

Идея:

- проверить ещё более короткий `max-model-len`
- если реальные тексты короткие, `8192` может быть завышен
- хороший кандидат, если latency падает без потери throughput

**GPU pod**

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-batched-tokens 131072 \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling
```

**CPU pod**

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 7m \
  --csv=/tmp/len2048-tokens131k \
  --html=/tmp/len2048-tokens131k.html \
  --csv-full-history
```

### Config 3: `len4096-tokens65k`

Идея:

- оставить умеренный `max-model-len=4096`
- снизить token budget до `65k`
- проверить, улучшатся ли average latency, p95 и p99

**GPU pod**

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 65536 \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling
```

**CPU pod**

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 7m \
  --csv=/tmp/len4096-tokens65k \
  --html=/tmp/len4096-tokens65k.html \
  --csv-full-history
```

### Config 4: `len2048-tokens65k`

Идея:

- компактный основной production-like кандидат
- меньше `max-model-len`
- меньше `max-num-batched-tokens`
- потенциально лучший баланс throughput / latency

**GPU pod**

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-batched-tokens 65536 \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling
```

**CPU pod**

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 7m \
  --csv=/tmp/len2048-tokens65k \
  --html=/tmp/len2048-tokens65k.html \
  --csv-full-history
```

### Config 5: `len2048-tokens49k`

Идея:

- fallback на ещё меньший token budget
- нужен, если `65k` всё ещё даёт высокий p95/p99
- latency-oriented кандидат

**GPU pod**

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-batched-tokens 49152 \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling
```

**CPU pod**

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 7m \
  --csv=/tmp/len2048-tokens49k \
  --html=/tmp/len2048-tokens49k.html \
  --csv-full-history
```

---

## Диагностическая волна: Config 6-7

Эти конфиги запускаются **только после выбора лучшей базы** из Config 1-5.

### Config 6: `best-no-chunked-prefill`

Идея:

- для encoder/pooling workload chunked prefill может быть не нужен
- проверяем, станет ли latency стабильнее
- особенно смотрим `p95` и `p99`

Ниже пример для базы `len2048-tokens131k`.  
Если победил другой конфиг, подставь его `max-model-len` и `max-num-batched-tokens`.

**GPU pod**

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-batched-tokens 131072 \
  --no-enable-chunked-prefill \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling
```

**CPU pod**

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 7m \
  --csv=/tmp/best-no-chunked-prefill-len2048-tokens131k \
  --html=/tmp/best-no-chunked-prefill.html \
  --csv-full-history
```

### Config 7: `best-enforce-eager`

Идея:

- проверить CUDA graphs vs eager
- обычно eager медленнее
- но для нестандартного plugin/pooling workload иногда даёт более предсказуемую latency

Ниже пример для базы `len2048-tokens131k`.  
Если победил другой конфиг, подставь его значения.

**GPU pod**

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 4 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --max-num-batched-tokens 131072 \
  --enforce-eager \
  --io-processor-plugin mmbert_gliner2_io \
  -- --runner pooling
```

**CPU pod**

```bash
cd ~/gliner-guard-serve/test-script

GLINER_HOST=http://<POD_ID>.runpod.internal:8000 \
python -m locust \
  -f test-gliner-vllm.py \
  --headless \
  -u 100 \
  -r 1 \
  --run-time 7m \
  --csv=/tmp/best-enforce-eager-len2048-tokens131k \
  --html=/tmp/best-enforce-eager-len2048-tokens131k.html \
  --csv-full-history
```

---

## Рекомендуемый порядок запуска

### Основная волна

```text
0. baseline-len8192-tokens262k
1. len4096-tokens131k
2. len2048-tokens131k
3. len4096-tokens65k
4. len2048-tokens65k
5. len2048-tokens49k
```

### Диагностическая волна

После выбора лучшего base-конфига:

```text
6. best-no-chunked-prefill
7. best-enforce-eager
```

---

## Что сохранять после каждого прогона

Минимум:

```text
Requests/s
Average Response Time
p50
p95
p99
Failure Count
Exceptions
GPU util
VRAM
```

Если используешь GPU wrappers, сохраняй ещё:

```text
/tmp/vllm-gpu-traces/<experiment>.gpu.csv
/tmp/vllm-gpu-traces/<experiment>.vllm.log
```

Плохой конфиг — это не только OOM.

Признаки плохого конфига:

- throughput не растёт или падает
- average latency выросла
- p95/p99 резко ухудшились
- появились failures
- появились exceptions
- GPU util высокий, но полезного throughput больше нет

---

## Как принимать решение

### Шаг 1. Проверить, нужен ли `max-model-len=8192`

Сравнить:

```text
baseline-len8192-tokens262k
len4096-tokens131k
len2048-tokens131k
```

Если `4096` или `2048` дают тот же throughput, но ниже latency, значит `8192` был завышен.

### Шаг 2. Проверить меньший token budget

Сравнить:

```text
len4096-tokens65k
len2048-tokens65k
len2048-tokens49k
```

Если `65k` или `49k` дают почти тот же throughput, но ниже `p95/p99`, они лучше для production-like режима.

### Шаг 3. Сделать A/B на лучшем конфиге

Сравнить:

```text
best
best-no-chunked-prefill
best-enforce-eager
```

Интерпретация:

- если `best-no-chunked-prefill` лучше по `p95/p99`, оставить `--no-enable-chunked-prefill`
- если `best-enforce-eager` хуже, оставить CUDA graphs
- если `best-enforce-eager` лучше или стабильнее, можно рассмотреть eager

---

## Что сейчас не трогаем

Пока не смешиваем новые оси:

```text
--gpu-memory-utilization
--max-num-seqs
--cuda-graph-sizes
--async-scheduling
```

Почему:

- иначе трудно понять, что именно дало эффект
- сейчас главные подозреваемые: `max-model-len` и `max-num-batched-tokens`
- GLiNER2/mBERT pooling больше похож на encoder/prefill-heavy workload, чем на классический decode-heavy LLM serving

---

## Troubleshooting

| Проблема | Что делать |
|---|---|
| `vllm-factory-serve: command not found` | активировать правильный venv, проверить `python -m pip show vllm-factory` |
| `Server failed to start` | смотреть stdout/stderr сервера и `vllm.log` |
| `Connection refused` с CPU pod | сервер ещё не стартовал или сервис слушает не `0.0.0.0` |
| `getent hosts <POD_ID>.runpod.internal` пустой | `Global Networking` не включён или pod был пересоздан |
| Locust даёт `0 req/s` | проверить `curl http://<POD_ID>.runpod.internal:8000/health` |
| OOM или unhealthy start | уменьшить `max-model-len` или `max-num-batched-tokens` |

---

## Короткий итог

Осмысленная матрица:

```text
baseline-len8192-tokens262k
len4096-tokens131k
len2048-tokens131k
len4096-tokens65k
len2048-tokens65k
len2048-tokens49k
best-no-chunked-prefill
best-enforce-eager
```

Эта матрица честно соответствует ограничениям:

```text
Locust fixed
max-batch-size fixed
num-instances fixed
GLiNER2/mBERT pooling workload
```

Главный вопрос, на который она отвечает:

```text
Какой max-model-len и token budget дают лучший throughput/latency при фиксированной нагрузке и фиксированном batch-size?
```
