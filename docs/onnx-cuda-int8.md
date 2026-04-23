# ONNX CUDA INT8 (локальный бандл)

Инференс `gliner2-multi-v1` через LitServe с ONNX Runtime (CUDA EP) и INT8-моделью, экспортированной локально.

## Особенности

- **Модель:** `fastino/gliner2-multi-v1` (экспорт в ONNX через `gliner2-onnx`)
- **Runtime:** ONNX Runtime (`CUDAExecutionProvider`, fallback `CPUExecutionProvider`)
- **Точность:** `int8`
- **Требования:** NVIDIA GPU (CUDA 12+), отдельный CPU pod/VM для Locust
- **Batching:** dynamic batching, `max_batch_size=64`, `batch_timeout=0.05s`
- **Workers:** 4 воркера на устройство (`workers_per_device=4`)
- **Fast queue:** включен

## Важный контекст

Готовая модель `cuerbot/gliner2-multi-v1` не подошла под текущий `GLiNER2ONNXRuntime.from_pretrained(...)` в этом проекте (нет `gliner2_config.json`).

Поэтому использовался путь:
1. локально экспортировать `int8`-бандл через `gliner2-onnx`;
2. загрузить его из директории (локальный путь), а не из HF repo id.

Для этого в `onnx/main.py` добавлена поддержка локального пути в `ONNX_MODEL_NAME`.

## Запуск

### 1. Экспорт INT8-модели (GPU pod)

```bash
cd /workspace
git clone https://github.com/bogdanminko/gliner2-onnx.git
cd gliner2-onnx
uv sync
make onnx-export MODEL=fastino/gliner2-multi-v1 QUANTIZE=int8
```

Ожидаемый результат:
`/workspace/gliner2-onnx/model_out/gliner2-multi-v1`

### 2. Поднять сервер LitServe на локальном INT8-бандле (GPU pod)

```bash
cd /workspace/gliner-guard-serve/onnx

export ONNX_MODEL_NAME=/workspace/gliner2-onnx/model_out/gliner2-multi-v1
export ONNX_PRECISION=int8
export ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
unset ONNX_PROVIDER_OPTIONS
export NUM_WORKERS=4
export MAX_BATCH_SIZE=64

uv run main.py
```

Сервер стартует на `http://0.0.0.0:8000`.

### 3. Smoke test (с CPU pod или с GPU pod)

```bash
curl -X POST "https://<gpu-pod-id>-8000.proxy.runpod.net/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Send $500 to John Smith at john.smith@gmail.com"}'
```

Пример валидного ответа:

```json
{"entities":[],"classifications":{"safety":{"unsafe":0.5075716376304626}}}
```

### 4. Нагрузочный тест Locust (CPU pod)

```bash
cd /workspace/gliner-guard-serve/test-script
uv sync

export GLINER_HOST="https://<gpu-pod-id>-8000.proxy.runpod.net"

uv run locust -f test-gliner.py --headless -u 100 -r 1 --run-time 15m \
  --csv=stats_gliner2_multi_onnx_cuda_int8 \
  --html stats_gliner2_multi_onnx_cuda_int8.html
```

### 5. Сохранить результаты

Скопировать `stats_gliner2_multi_onnx_cuda_int8_stats.csv` в:

`results/litserve/gliner2-multi-v1/onnx_cuda_int8.csv`

При необходимости сохранить и HTML-отчет рядом.

## Результат эксперимента

Файл результата:

- `results/litserve/gliner2-multi-v1/onnx_cuda_int8.csv`

Таблица в `README.md` обновляется командой:

```bash
make bench-readme
```

## Примечания

- При высокой нагрузке (`u=100`) наблюдались частые `504 Gateway Timeout` из-за таймаута очереди LitServe (`timeout=30` по умолчанию).
- Это выражается в очень высоком `Err rate` и низком эффективном RPS для текущей конфигурации INT8.
- Для повторного прогона стоит начинать с меньшей нагрузки (`u=20/40/60`) и только потом выходить на `u=100`.
