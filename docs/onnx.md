# ONNX + TensorRT

Реализация инференса GLiNER Guard через LitServe с ONNX Runtime и TensorRT. Использует квантизованные ONNX-модели вместо PyTorch.

## Особенности

- **Модель:** `hivetrace/gliner-guard-uniencoder-onnx` (GLiNER2 ONNX)
- **Точность:** fp16 (int8 несовместим с TRT — см. ниже)
- **Требования:** NVIDIA GPU с CUDA 12+
- **Batching:** dynamic batching, `max_batch_size=64`, `batch_timeout=0.05s`
- **Workers:** 4 воркера на устройство (`workers_per_device=4`)
- **Fast queue:** включён

### Схема модели

- **PII-сущности:** person, address, email, phone (threshold 0.5)
- **Классификация:** safety (safe / unsafe)

## Запуск

### 1. Установка зависимостей

```bash
cd onnx
uv sync
```

### 2. TensorRT (обязательно)

```bash
uv add tensorrt
```

Зарегистрировать библиотеки TRT в системе (один раз, сохраняется после перезагрузки):

```bash
TENSORRT_LIBS=$(python -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))")
echo "$TENSORRT_LIBS" > /etc/ld.so.conf.d/tensorrt.conf
ldconfig
```

Создать директорию для кэшей:

```bash
mkdir -p trt_cache
```

### 3. Конфигурация `.env`

```bash
NUM_WORKERS=4
ONNX_PRECISION=fp16
ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider
ONNX_PROVIDER_OPTIONS=[{"trt_timing_cache_enable": "true", "trt_timing_cache_path": "trt_cache", "trt_force_timing_cache": "true", "trt_engine_cache_enable": "true", "trt_engine_cache_path": "trt_cache", "trt_dump_ep_context_model": "true", "trt_ep_context_file_path": "trt_cache"}, {}]
MAX_BATCH_SIZE=64
```

### 4. Запуск

```bash
uv run main.py
```

Сервер стартует на `http://localhost:8000`.

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Send $500 to John Smith at john.smith@gmail.com"}'
```

## TensorRT кэш

TRT компилирует движок при первом запросе для каждого уникального input shape — это медленно. После прогрева всех shapes кэш сохраняется на диск и последующие запуски быстрые.

| Кэш | Время старта |
|---|---|
| Без кэша | ~384s |
| Timing cache | ~42s |
| Engine cache | ~9s |
| Embedded engine | ~1.9s |

## Примечания

- **int8 не работает с TRT** — ONNX int8 модели используют асимметричную квантизацию (ненулевые zero points), которую TRT не поддерживает. Используй `fp16`.
- **`ONNX_PROVIDER_OPTIONS`** должен содержать ровно столько элементов, сколько провайдеров реально передаётся в ONNX Runtime. Если TRT доступен — библиотека использует 2 провайдера (TRT + CUDA), поэтому передаём 2 элемента: `[{...trt opts...}, {}]`.
- **Первые запросы под нагрузкой** будут медленными пока TRT компилирует движки для новых input shapes. После прогрева — быстро.