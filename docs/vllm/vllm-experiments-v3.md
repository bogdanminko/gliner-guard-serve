# vLLM v3: meaningful configs

Третья серия ручных экспериментов для GLiNER Guard через vLLM.

Этот документ описывает уже не "все возможные цифры подряд", а короткую
осмысленную матрицу из single-instance и multi-instance гипотез, которые реально
имеет смысл сравнивать между собой.

---

## Что исследуем

Главный вопрос этой волны:

```text
Какие конфиги действительно различаются по стратегии serving,
а не просто слегка двигают числа без новой гипотезы?
```

Матрица ниже построена как набор разных сценариев:

1. `single-safe` - стабильный single-instance baseline
2. `single-dense` - более плотный single-instance режим
3. `single-short` - гипотеза для коротких текстов
4. `multi-2x` - мягкий multi-instance вариант
5. `multi-4x` - агрессивный multi-instance вариант

Почему матрица выглядит именно так:

- для encoder/pooling serving ключевой смысл обычно дают `max-model-len`,
  `max-num-seqs` и стратегия single-instance vs multi-instance
- `max-num-batched-tokens` здесь скорее технический лимит, а не главная ось
  исследования
- для GLiNER2/ModernBERT важнее сравнить разные типы serving-поведения, чем
  бесконечно перебирать близкие scheduler-значения

Все конфиги ниже:

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

---

## Матрица экспериментов

| Config | Имя | Instances | max-model-len | max-num-seqs | max-num-batched-tokens | Идея |
|---|---|:---:|---:|---:|---:|---|
| 1 | `single-safe` | 1 | 8192 | 64 | 65536 | безопасный baseline |
| 2 | `single-dense` | 1 | 8192 | 128 | 65536 | плотный single-instance |
| 3 | `single-short` | 1 | 4096 | 128 | 65536 | короткие тексты |
| 4 | `multi-2x` | 2 | 8192 | n/a | 65536 | мягкий multi-instance |
| 5 | `multi-4x` | 4 | 8192 | n/a | 65536 | агрессивный multi-instance |

### Почему это не `dense-192 / dense-160 / dense-128`

Такие промежуточные варианты часто выглядят как "тонкая настройка", но на
первой осмысленной итерации они редко отвечают на новый инженерный вопрос.

Здесь мы специально оставляем только конфиги, где отличается сама гипотеза:

- безопасный vs плотный single-instance
- длинный vs короткий `max-model-len`
- single-instance vs multi-instance

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

Дождаться `Application startup complete`, затем остановить процесс `Ctrl+C`.

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

---

## Pre-flight перед полной матрицей

### Шаг 1. Поднять тестовый сервер на GPU pod

```bash
cd ~/gliner-guard-serve/vllm
./serve.sh
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
  --csv=/tmp/preflight-v3
```

### Шаг 5. Остановить тестовый сервер

На GPU pod: `Ctrl+C`.

---

## Общий цикл для любого конфига

### 1. На GPU pod поднять сервер с нужными флагами

Запускается только один конфиг за раз.

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

На CPU pod должны появиться:

- `/tmp/<experiment-name>_stats.csv`
- `/tmp/<experiment-name>_stats_history.csv`
- `/tmp/<experiment-name>_failures.csv`
- `/tmp/<experiment-name>_exceptions.csv`
- `/tmp/<experiment-name>.html`

### 4. Остановить сервер на GPU pod

Если процесс запущен в foreground, просто `Ctrl+C`.

### 5. Проверить, что порт освободился

```bash
ss -ltnp | grep ':8000'
```

### 6. Опционально: писать GPU trace

Если нужен параллельный GPU-мониторинг, можно использовать обёртки из
`docs/vllm_gliner2_mbert_gpu_tracking_wrappers.md`.

---

## Config 1: `single-safe`

Идея:

- стабильный single-instance baseline
- минимум риска по OOM и scheduler stalls

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
- пробуем выжать больше из одного процесса без перехода в multi-instance

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

- отдельная гипотеза про короткие тексты
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
- мягкий вариант перед `multi-4x`
- полезно как более production-like multi-instance baseline

Важно:

- это не `vllm serve`
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
- кандидат на максимум throughput, если GPU недогружен compute-wise

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

## Рекомендуемый порядок запуска

1. `single-safe`
2. `single-dense`
3. `single-short`
4. `multi-2x`
5. `multi-4x`

Такой порядок идёт от самого стабильного single-instance baseline к более
агрессивным стратегиям.

---

## Что сохранять после каждого прогона

Минимум:

- `*_stats.csv`
- `*_stats_history.csv`
- `*.html`

Если нужен более глубокий разбор:

- `*_failures.csv`
- `*_exceptions.csv`
- GPU trace
- лог сервера

---

## Как принимать решение

### Шаг 1. Отбросить нестабильные конфиги

Сразу убрать варианты, где:

- есть failures
- видны таймауты или ошибки endpoint
- p95/p99 заметно хуже без выигрыша по throughput

### Шаг 2. Сравнить single-instance варианты

Сначала понять, есть ли явный победитель среди:

- `single-safe`
- `single-dense`
- `single-short`

### Шаг 3. Сравнить победителя single-instance с multi-instance

Дальше уже сравнивать лучший single-instance baseline против:

- `multi-2x`
- `multi-4x`

Смысл этой волны именно в том, чтобы понять, нужен ли multi-instance вообще.

---

## Что сейчас не трогаем

В этой матрице специально не крутим:

- `enforce-eager`
- chunked prefill как отдельную ось
- десятки промежуточных `max-num-seqs`
- тонкий перезапуск c другими `gpu-memory-utilization`

Это отдельные диагностические шаги, а не цель `v3`.

---

## Troubleshooting

- Если новый конфиг не стартует, почти всегда причина в том, что старый процесс
  всё ещё держит порт `8000`.
- Если `multi-2x` или `multi-4x` стартуют нестабильно, сначала проверь, что тот
  же артефакт нормально работает через обычный single-instance `vllm serve`.
- Если CPU pod не достаёт до GPU pod, проверь `Global Networking` и актуальный
  internal DNS.

---

## Короткий итог

`v3` - это сокращённая, осмысленная матрица. Она нужна, чтобы сравнивать не
случайные соседние настройки, а разные реальные стратегии serving:

1. безопасный single-instance
2. плотный single-instance
3. short-context single-instance
4. мягкий multi-instance
5. агрессивный multi-instance
