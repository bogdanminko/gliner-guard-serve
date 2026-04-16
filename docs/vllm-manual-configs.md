# vLLM manual runbook: all configs

Этот файл описывает ручной запуск всех vLLM-конфигов для стенда:
- **GPU pod** поднимает `vllm serve` или `vllm-factory-serve`
- **CPU pod** запускает `Locust`
- связь между pod идет через `Global Networking`

Базовый host GPU pod в примерах:

```text
<POD_ID>.runpod.internal
```

Если GPU pod менялся, подставь свой актуальный `POD_ID.runpod.internal`.

Все конфиги ниже: `float16` + CUDA graphs. 

## Короткий ответ на главный вопрос

**Да, перед запуском следующего конфига предыдущий сервер нужно остановить.**

Почему:
- все конфиги слушают один и тот же порт `8000`
- флаги scheduler применяются только при старте процесса
- если оставить старый `./serve.sh` или старый `vllm serve`, новый конфиг не поднимется или поднимется не тот процесс

Что именно нужно делать:
1. Запустить один конфиг на **GPU pod**
2. Прогнать `Locust` с **CPU pod**
3. Дождаться завершения `Locust`
4. **Остановить текущий сервер на GPU pod** (`Ctrl+C`)
5. Только потом запускать следующий конфиг

Если раньше был запущен `./serve.sh`, его тоже нужно остановить перед ручным запуском конфигов.

## Общий порядок для каждого конфига

### 1. На GPU pod

Запустить сервер с нужными флагами.

### 2. На CPU pod

Запустить `Locust`:

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

### 3. После завершения

На CPU pod будут файлы:
- `/tmp/<experiment-name>_stats.csv`
- `/tmp/<experiment-name>_stats_history.csv`
- `/tmp/<experiment-name>_failures.csv`
- `/tmp/<experiment-name>_exceptions.csv`
- `/tmp/<experiment-name>.html`

### 4. Остановить сервер на GPU pod

Если сервер запущен в foreground, просто:

```bash
Ctrl+C
```

Если не уверен, что старый процесс остался, сначала проверь:

```bash
ss -ltnp | grep ':8000'
```

## Config 1: `sched-safe`

Идея:
- консервативный scheduler
- меньше memory pressure
- хороший baseline для начала

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

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

## Config 2: `sched-balanced`

Идея:
- умеренная конкурентность
- основной кандидат на best-effort production config

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm serve /tmp/gliner-guard-uni-vllm \
  --runner pooling \
  --trust-remote-code \
  --dtype float16 \
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

## Config 3: `sched-aggressive`

Идея:
- максимальный batching
- самый рискованный по latency / memory pressure

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm serve /tmp/gliner-guard-uni-vllm \
  --runner pooling \
  --trust-remote-code \
  --dtype float16 \
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

## Config 4: `sched-short`

Идея:
- если тексты короткие
- уменьшенный `max-model-len`
- больше потенциальной плотности по seqs

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm serve /tmp/gliner-guard-uni-vllm \
  --runner pooling \
  --trust-remote-code \
  --dtype float16 \
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

## Config 5: `multi-4x`

Идея:
- 4 instance на одном GPU
- reverse proxy от `vllm-factory-serve`
- хороший тест для memory-bound encoder serving

Важно:
- это **не** `vllm serve`
- это `vllm-factory-serve`
- предыдущий single-instance сервер должен быть полностью остановлен

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

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

## Рекомендуемый порядок прогона

1. `sched-safe`
2. `sched-balanced`
3. `sched-aggressive`
4. `sched-short`
5. `multi-4x`

## Что проверять после каждого прогона

- `Failure Count = 0`
- есть `Requests/s`
- есть `Average Response Time`
- смотреть `95%` и `99%`
- HTML-отчёт сохранился

## Минимальный чек-лист между конфигами

После завершения каждого прогона:

1. На CPU pod убедиться, что `Locust` завершился
2. На GPU pod остановить сервер (`Ctrl+C`)
3. Проверить, что порт освободился:

```bash
ss -ltnp | grep ':8000'
```

Если вывода нет — можно запускать следующий конфиг.

## Если хочешь использовать `./serve.sh`

Для ручного прогона матрицы **не надо** использовать `./serve.sh` перед каждым конфигом.

Почему:
- `./serve.sh` запускает default single-instance baseline (`float16` + CUDA graphs)
- твои эксперименты требуют явных scheduler flags
- эти флаги удобнее и правильнее задавать прямой командой `vllm serve` / `vllm-factory-serve`

`./serve.sh` нужен только:
- для первичной подготовки модели
- для быстрого smoke test
- для проверки, что endpoint в принципе поднимается
