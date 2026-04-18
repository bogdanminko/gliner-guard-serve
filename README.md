# Gliner Guard Serve

## Тестовые данные
Лежат в директории :`test-script` в файлах `prompts.csv` (`user_msg`) и `responses.csv` (`assistant_msg`) — по 500 строк синтетического текста с шумом. Длина 128–512 слов, среднее ~320.

### Статистика по словам

| файл | rows | min words | max words | avg words |
|------|------|-----------|-----------|-----------|
| prompts.csv | 500 | 128 | 512 | ~320 |
| responses.csv | 500 | 128 | 512 | ~320 |

### Статистика по символам

| файл | rows | min chars | max chars | avg chars | median | stdev |
|------|------|-----------|-----------|-----------|--------|-------|
| prompts.csv | 500 | 914 | 4 152 | 2 521 | 2 472 | 702 |
| responses.csv | 500 | 949 | 4 139 | 2 534 | 2 507 | 737 |

## Benchmarks

Пайплайн бенчмаркинга, настройка стенда и инструкция по тестированию — [docs/instruction.md](docs/instruction.md).
Итоговый sweep по dynamic batching на Runpod A100 сохранён в [docs/ray-serve-sweep-2026-04-18-final.md](docs/ray-serve-sweep-2026-04-18-final.md).

Таблица обновляется командой: `make bench-readme`
считывая curated `.csv` файлы рекурсивно из директории `results/`
Ray Serve прогоны теперь можно повторять в обоих режимах: `TORCH_DTYPE=bf16` и `TORCH_DTYPE=fp16`.

<!-- BENCH:START -->
| Model                         | Serving   | Runtime                    |     RPS | P50 (ms) | P95 (ms) | P99 (ms) | Err rate (%) |
| ----------------------------- | --------- | -------------------------- | ------: | -------: | -------: | -------: | -------: |
| gliner-guard-uni              | litserve  | onnx-cuda-fp16             |   170.6 |      540 |      870 |     1000 |     0.00 |
|                               |           | onnx-trt-fp16              | **193.6** |      480 |      750 |      900 |     0.00 |
|                               |           | torch-fp16-flash-attn      |   157.0 |      550 |     1200 |     1400 |     0.00 |
|                               |           | torch-fp16-spda            |   159.4 |      540 |     1200 |     1400 |     0.00 |
| gliner2-multi-v1              | litserve  | pytorch-fp16               |    83.7 |     1200 |     2000 |     2400 |    12.95 |
|                               |           | onnx-cuda-fp16             |    90.8 |     1000 |     1700 |     2100 |     0.00 |
|                               |           | onnx-trt-fp16              |   122.6 |      740 |     1200 |     1400 |     0.00 |
| gliner2-multi-v1-flashdeberta | litserve  | pytorch-fp16-flashdeberta  |   105.7 |      950 |     1400 |     1700 |     0.00 |
|                               |           |                            |         |          |          |          |          |
| gliner-guard-bi               | ray-serve | pytorch-fp16-rest-nobatch  |    22.0 |     4400 |     5000 |     5200 |     0.00 |
|                               |           | pytorch-fp16-rest-dynbatch |    68.3 |     1500 |     1800 |     2000 |     0.00 |
|                               |           | pytorch-fp16-grpc-dynbatch |    17.7 |       56 |       60 |       65 |     0.00 |
|                               |           | pytorch-bf16-rest-nobatch  |    22.3 |     4500 |     4700 |     4800 |     0.00 |
|                               |           | pytorch-bf16-rest-dynbatch |    67.0 |     1400 |     1900 |     2200 |     0.00 |
|                               |           | pytorch-bf16-grpc-dynbatch |    17.4 |       56 |       62 |       67 |     0.00 |
| gliner-guard-uni              | ray-serve | pytorch-fp16-rest-nobatch  |    34.1 |     2900 |     3200 |     3600 |     0.00 |
|                               |           | pytorch-fp16-rest-dynbatch |    71.1 |     1400 |     1600 |     1700 |     0.00 |
|                               |           | pytorch-fp16-grpc-dynbatch |    23.5 |   **41** |       47 |       51 |     0.00 |
|                               |           | pytorch-bf16-rest-nobatch  |    33.5 |     3000 |     3200 |     3300 |     0.00 |
|                               |           | pytorch-bf16-rest-dynbatch |    66.1 |     1500 |     1700 |     1800 |     0.00 |
|                               |           | pytorch-bf16-grpc-dynbatch |    23.3 |       42 |   **46** |   **48** |     0.00 |
<!-- BENCH:END -->
