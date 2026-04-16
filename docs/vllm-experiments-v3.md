# vLLM manual runbook: meaningful configs

Этот файл описывает ручной запуск **осмысленной** матрицы vLLM-конфигов для стенда:
- **GPU pod** поднимает `vllm serve` или `vllm-factory-serve`
- **CPU pod** запускает `Locust`
- связь между pod идет через `Global Networking`

Базовый host GPU pod в примерах:

```text
<POD_ID>.runpod.internal
```

Если GPU pod менялся, подставь свой актуальный `POD_ID.runpod.internal`.

Все конфиги ниже: `dtype auto` + CUDA graphs.

---

## Главная идея этой матрицы

Матрица ниже построена не как просто набор разных цифр, а как набор **разных гипотез**:

1. **single-safe** — стабильный baseline с низким риском memory pressure
2. **single-dense** — более плотный single-instance режим
3. **single-short** — отдельная гипотеза для коротких текстов
4. **multi-2x** — мягкий multi-instance режим
5. **multi-4x** — агрессивный multi-instance режим

Почему именно так:
- для encoder/pooling serving главный смысл обычно несут `max-model-len`, `max-num-seqs` и сама стратегия single-instance vs multi-instance
- для encoder-only сценариев `max-num-batched-tokens` обычно менее полезен как тонкий scheduler knob, чем для decoder LLM, поэтому здесь он используется скорее как большой технический лимит, а не как основная ось матрицы
- `vllm-factory-serve` особенно интересен для encoder workload, где один процесс может плохо утилизировать GPU по compute

---

## Короткий ответ на главный вопрос

**Да, перед запуском следующего конфига предыдущий сервер нужно остановить.**

Почему:
- все конфиги слушают один и тот же порт `8000`
- флаги scheduler применяются только при старте процесса
- если оставить старый `vllm serve`, `vllm-factory-serve` или `./serve.sh`, новый конфиг не поднимется или поднимется не тот процесс

Что именно нужно делать:
1. Запустить один конфиг на **GPU pod**
2. Прогнать `Locust` с **CPU pod**
3. Дождаться завершения `Locust`
4. **Остановить текущий сервер на GPU pod** (`Ctrl+C`)
5. Только потом запускать следующий конфиг

Если раньше был запущен `./serve.sh`, его тоже нужно остановить перед ручным запуском конфигов.

---

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

Если вывода нет — можно запускать следующий конфиг.

---

## Config 1: `single-safe`

Идея:
- стабильный baseline
- минимум риска по OOM / scheduler stalls
- точка отсчёта для всех следующих прогонов

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm serve /tmp/gliner-guard-uni-vllm \
  --runner pooling \
  --trust-remote-code \
  --dtype auto \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.75 \
  --io-processor-plugin mmbert_gliner2_io \
  --max-model-len 8192 \
  --max-num-seqs 64 \
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
  --csv=/tmp/single-safe \
  --html=/tmp/single-safe.html \
  --csv-full-history
```

---

## Config 2: `single-dense`

Идея:
- основной кандидат на лучший single-instance throughput
- проверяем, сколько можно выжать из одного процесса без multi-instance

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
  --csv=/tmp/single-dense \
  --html=/tmp/single-dense.html \
  --csv-full-history
```

---

## Config 3: `single-short`

Идея:
- отдельная гипотеза: реальные тексты короткие
- уменьшаем `max-model-len`, чтобы освободить память и поднять плотность serving

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
  --max-num-seqs 128 \
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
  --csv=/tmp/single-short \
  --html=/tmp/single-short.html \
  --csv-full-history
```

---

## Config 4: `multi-2x`

Идея:
- первая проверка multi-instance стратегии
- не сразу 4 инстанса, а мягкий вариант
- полезно как production-like safer multi-instance

Важно:
- это **не** `vllm serve`
- это `vllm-factory-serve`
- предыдущий single-instance сервер должен быть полностью остановлен

### GPU pod

```bash
cd ~/gliner-guard-serve/vllm

vllm-factory-serve /tmp/gliner-guard-uni-vllm \
  --num-instances 2 \
  --max-batch-size 64 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-batched-tokens 65536 \
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
  --csv=/tmp/multi-2x \
  --html=/tmp/multi-2x.html \
  --csv-full-history
```

---

## Config 5: `multi-4x`

Идея:
- агрессивный multi-instance режим
- главный кандидат на максимум throughput, если GPU действительно недогружен compute-wise

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
  --dtype auto \
  --max-model-len 8192 \
  --max-num-batched-tokens 65536 \
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

## Рекомендуемый порядок прогона

1. `single-safe`
2. `single-dense`
3. `single-short`
4. `multi-2x`
5. `multi-4x`

---

## Что проверять после каждого прогона

- `Failure Count = 0`
- есть `Requests/s`
- есть `Average Response Time`
- смотреть `95%` и `99%`
- HTML-отчёт сохранился

Но для выбора победителя важен не только throughput.

Порядок принятия решения:
1. Сначала отбрасываешь всё, где есть failures или явная нестабильность
2. Потом сравниваешь `Requests/s`, `Average Response Time`, `p95`, `p99`
3. Победитель — либо максимум throughput при нормальном p95/p99, либо лучший компромисс для production-like режима

---

## Минимальный чек-лист между конфигами

После завершения каждого прогона:

1. На CPU pod убедиться, что `Locust` завершился
2. На GPU pod остановить сервер (`Ctrl+C`)
3. Проверить, что порт освободился:

```bash
ss -ltnp | grep ':8000'
```

Если вывода нет — можно запускать следующий конфиг.

---

## Если хочешь использовать `./serve.sh`

Для ручного прогона матрицы **не надо** использовать `./serve.sh` перед каждым конфигом.

Почему:
- `./serve.sh` запускает default single-instance baseline (`dtype auto` + CUDA graphs)
- эксперименты выше требуют явных параметров serving
- эти параметры удобнее и правильнее задавать прямой командой `vllm serve` / `vllm-factory-serve`

`./serve.sh` нужен только:
- для первичной подготовки модели
- для быстрого smoke test
- для проверки, что endpoint в принципе поднимается

---

## Минимальный вариант матрицы

Если нужен не полный набор, а короткий и практичный прогон, можно оставить только 4 конфига:

1. `single-safe`
2. `single-dense`
3. `single-short`
4. `multi-4x`

`multi-2x` в этом случае можно пропустить.
