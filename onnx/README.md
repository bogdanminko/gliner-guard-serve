# ONNX + TensorRT

LitServe inference server с ONNX Runtime и TensorRT.

Подробная документация: [docs/onnx-trt.md](../docs/onnx-trt.md)

## Запуск

```bash
uv sync
uv run main.py
```

## Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Send $500 to John Smith at john.smith@gmail.com"}'
```