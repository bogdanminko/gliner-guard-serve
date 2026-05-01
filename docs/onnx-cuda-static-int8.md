# ONNX CUDA static INT8 (gliner2-multi-v1)

Инференс `fastino/gliner2-multi-v1` через LitServe и ONNX Runtime (CUDA EP) с **статически** квантизованными ONNX-моделями (ORT `quantize_static`, QDQ). Отдельно от динамического int8 см. [onnx-cuda-int8.md](onnx-cuda-int8.md).

## Особенности

- **Экспорт fp32:** `gliner2-onnx` → `tools/export_model.py` (encoder, classifier, span_rep; `count_embed` отдельно).
- **Квантизация:** `onnxruntime.quantization.quantize_static` для каждого `.onnx` (после экспорта появляются `*_int8.onnx`).
- **Конфиг:** в `gliner2_config.json` в `onnx_files` добавлен блок `int8` со всеми четырьмя компонентами (включая `count_embed`).
- **Токенизатор:** в `tokenizer_config.json` / `config.json` локального бандла класс `TokenizersBackend` заменён на `DebertaV2TokenizerFast`, иначе `transformers.AutoTokenizer` не загружается.
- **Сервер:** `onnx/main.py` — `ONNX_MODEL_NAME` указывает на **локальную директорию** бандла; `ONNX_PRECISION=int8`.
- **Нагрузка (зафиксированный прогон в таблице):** Locust с ноутбука, `u=20`, `r=1`, длительность по факту прогона; префикс CSV `stats_gliner2_multi_onnx_static_int8_u20`. Артефакты лежат в `results/litserve/gliner2-multi-v1/` под именами `onnx-cuda-static-int8-u20*`.

## Предпосылки на GPU pod

- Репозитории: `gliner2-onnx`, `gliner-guard-serve`.
- В `gliner2-onnx`: `uv sync`, зависимости для экспорта (`torch`, `onnx`, `gliner2` из git и т.д. по сообщениям окружения).

## 1. Экспорт fp32

```bash
cd /workspace/gliner2-onnx
uv run python tools/export_model.py --model fastino/gliner2-multi-v1
```

## 2. Экспорт count_embed

```bash
uv run python tools/export_count_embed.py --model fastino/gliner2-multi-v1 --save-path model_out/gliner2-multi-v1
```

## 3. Static INT8 для каждого компонента

Для каждого файла `onnx/*.onnx` без суффикса `_fp16` / `_int8` запустить `quantize_static` (калибровка — см. использованный у вас скрипт; для быстрого прогона допускается уменьшенное число сэмплов и `per_channel=False` на тяжёлом `encoder`).

В итоге в `model_out/gliner2-multi-v1/onnx/` должны быть как минимум:

- `encoder_int8.onnx`, `classifier_int8.onnx`, `span_rep_int8.onnx`, `count_embed_int8.onnx`

## 4. Обновить `gliner2_config.json`

В `onnx_files` добавить ключ `int8` с путями ко всем четырём `*_int8.onnx` (относительно корня бандла).

## 5. Исправить токенизатор

В каталоге бандла в `tokenizer_config.json` и при необходимости `config.json`:

- `tokenizer_class`: `DebertaV2TokenizerFast`
- удалить проблемные `auto_map`, если мешают загрузке.

Проверка:

```bash
uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('model_out/gliner2-multi-v1')"
```

## 6. Запуск LitServe

```bash
cd /workspace/gliner-guard-serve/onnx
export ONNX_MODEL_NAME=/workspace/gliner2-onnx/model_out/gliner2-multi-v1
export ONNX_PRECISION=int8
export ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
unset ONNX_PROVIDER_OPTIONS
export NUM_WORKERS=4
export MAX_BATCH_SIZE=64
uv run --no-sync main.py
```

Примечание: `uv run` без `--no-sync` может пересобрать `.venv` и откатить ручные версии пакетов.

## 7. Нагрузка и результаты

```bash
cd test-script
export GLINER_HOST="https://<gpu-pod>-8000.proxy.runpod.net"
uv run locust -f test-gliner.py --headless -u 20 -r 1 --run-time <длительность> \
  --csv=stats_gliner2_multi_onnx_static_int8_u20 \
  --html stats_gliner2_multi_onnx_static_int8_u20.html
```

Сохранить в репозиторий (основной файл для `make bench-readme`):

- `results/litserve/gliner2-multi-v1/onnx-cuda-static-int8-u20.csv` (строка `Aggregated` из `*_stats.csv`)

Опционально рядом: `*_failures.csv`, `*_exceptions.csv`, `*_stats_history.csv`, `*.html`.

Обновление таблицы в README:

```bash
make bench-readme
```

## Интерпретация текущих цифр

При высоком **Err rate** и перцентилях ~60s доминируют **таймауты клиента/очереди**, а не «чистая» латентность одного forward. Для сравнения с fp16 имеет смысл снижать `u`, увеличивать `timeout` в `LitServe` и убедиться, что доля ошибок близка к нулю.
