# LitServe Flash Attention

Базовая реализация инференса GLiNER Guard через LitServe. Без дополнительных оптимизаций — PyTorch + fp16.

## Особенности

- **Модель:** `hivetrace/gliner-guard-uniencoder` (GLiNER2)
- **Точность:** fp16
- **Flash Attention** версия 2.8.3
- **Требования:** NVIDIA GPU, MPS, CPU
- **Batching:** dynamic batching, `max_batch_size=64`, `batch_timeout=0.05s`
- **Workers:** 4 воркера на устройство (`workers_per_device=4`)
- **Fast queue:** включён

### Схема модели

- **PII-сущности:** person, address, email, phone (threshold 0.4)
- **Классификация:** safety (safe / unsafe)

## Запуск
```bash
cd litserve-baseline
uv sync
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
uv run main.py
```
Если вы все сделали правильно, то далее во время запуска вы можете увидеть следующее предупреждение:
```sh
Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in ModernBertModel is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", dtype=torch.float16)`
```
Все ок - это предупреждение от Flash Attn, подождите немного, а затем появится следующий лог:

```
============================================================
🧠 Model Configuration
============================================================
Encoder model      : bogdanminko/mmBERT-small
Counting layer     : count_lstm_v2
Token pooling      : first
Attention backend  : 'flash_attention_2'
Encoder dtype      : torch.float16
```
Это значит что все запустилось и dtype правильный

Сервер стартует на `http://localhost:8000`.

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Send $500 to John Smith at john.smith@gmail.com"}'
```

## Быстрый тест (без Locust)

```bash
uv run python bench.py
```

Отправляет 128 асинхронных запросов и выводит RPS / avg latency.