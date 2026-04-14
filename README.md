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

Таблица обновляется командой: `make bench-readme`
считывая `.csv` файлы из директории `results`

<!-- BENCH:START -->
| Model            | Serving  | Runtime                   |     RPS | P50 (ms) | P95 (ms) | P99 (ms) | Err rate (%) |
| ---------------- | -------- | ------------------------- | ------: | -------: | -------: | -------: | -------: |
| gliner-guard-uni | litserve | pytorch-fp16              |   148.2 |      570 |     1500 |     1700 |     0.00 |
|                  |          | onnx-cuda-fp16            |   170.6 |      540 |      870 |     1000 |     0.00 |
|                  |          | onnx-trt-fp16             | **193.6** |  **480** |  **750** |  **900** |     0.00 |
|                  |          | torch-fp16-flash-attn     |   157.0 |      550 |     1200 |     1400 |     0.00 |
|                  |          | torch-fp16-spda           |   159.4 |      540 |     1200 |     1400 |     0.00 |
| gliner2-multi    | litserve | pytorch-fp16              |    83.7 |     1200 |     2000 |     2400 |    12.95 |
|                  |          | onnx-cuda-fp16            |    90.8 |     1000 |     1700 |     2100 |     0.00 |
|                  |          | onnx-trt-fp16             |   122.6 |      740 |     1200 |     1400 |     0.00 |
|                  |          | pytorch-fp16-flashdeberta |   105.7 |      950 |     1400 |     1700 |     0.00 |
<!-- BENCH:END -->
