# LitServe + FlashDeBERTa (GLiNER2 Multi)

Инференс `fastino/gliner2-multi-v1` через LitServe с включенной FlashDeBERTa.

## Особенности

- **Модель:** `fastino/gliner2-multi-v1` (GLiNER2)
- **Runtime:** PyTorch + FlashDeBERTa
- **Точность:** fp16
- **Требования:** NVIDIA GPU (тестировалось на A100 SXM 80GB), CUDA 12+
- **Batching:** dynamic batching, `max_batch_size=64`, `batch_timeout=0.05s`
- **Workers:** 4 воркера на устройство (`workers_per_device=4`)
- **Fast queue:** включён

## Стенд

- **GPU Pod (RunPod):** сервер инференса (`litserve-baseline/main.py`), открыт HTTP-порт `8000`
- **CPU Pod / VM:** генерация нагрузки через Locust (`test-script/test-gliner.py`)

## Запуск сервера (GPU Pod)

```bash
git clone https://github.com/bogdanminko/gliner-guard-serve/tree/main
cd /workspace/gliner-guard-serve/litserve-baseline
pip install uv
uv sync
uv pip install flashdeberta
uv pip install "gliner2 @ git+https://github.com/fastino-ai/GLiNER2.git"
export USE_FLASHDEBERTA=1 # можно добавить в коде файла main.py
export TORCH_MODEL_NAME=fastino/gliner2-multi-v1 # можно добавить в коде файла main.py
uv run --no-sync main.py
```

Почему `--no-sync`: при обычном `uv run` lock-файл может откатить `gliner2` обратно на PyPI-версию без нужной ветки с FlashDeBERTa.

Ожидаемый признак в логах при старте:

```text
Using FlashDeberta backend.
```

Сервер доступен по URL RunPod для порта `8000`, например:
`https://<gpu-pod-id>-8000.proxy.runpod.net`

### Пример запроса

```bash
curl -X POST https://<gpu-pod-id>-8000.proxy.runpod.net/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Send $500 to John Smith at john.smith@gmail.com"}'
```

## Нагрузочное тестирование (Locust)

На CPU Pod:

```bash
git clone https://github.com/bogdanminko/gliner-guard-serve/tree/main
cd /workspace/gliner-guard-serve/test-script
pip install uv
uv sync
export GLINER_HOST="https://<gpu-pod-id>-8000.proxy.runpod.net"
uv run locust -f test-gliner.py --headless -u 100 -r 1 --run-time 15m \
  --csv=stats_gliner2_multi_flashdeberta \
  --html stats_gliner2_multi_flashdeberta.html
```

После завершения прогона (`run-time limit reached`) файлы появятся в текущей директории:

- `stats_gliner2_multi_flashdeberta_stats.csv`
- `stats_gliner2_multi_flashdeberta_stats_history.csv`
- `stats_gliner2_multi_flashdeberta_failures.csv`
- `stats_gliner2_multi_flashdeberta_exceptions.csv`
- `stats_gliner2_multi_flashdeberta.html` (если версия Locust поддерживает сохранение html в headless)

## Сохранение результатов в репозиторий

Для генерации таблицы в `README.md` нужен CSV со строкой `Aggregated` и колонками `Request Count`, `Requests/s`, `50%`, `95%`, `99%`.

Скопируйте итоговый файл в:

```text
results/litserve/gliner2-multi/pytorch-fp16-flashdeberta.csv
```

Если используете Locust 2.43+, таким файлом обычно является:
`stats_gliner2_multi_flashdeberta_stats.csv`.

Далее обновите таблицу:

```bash
make bench-readme
```
