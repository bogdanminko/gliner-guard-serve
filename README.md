# Gliner Guard Serve

## Тестовые данные

Генерация: `python test-script/generate_data.py`

Создаёт `prompts.csv` (`user_msg`) и `responses.csv` (`assistant_msg`) — по 500 строк синтетического текста с шумом. Длина ~256–1024 слов, среднее ~512.

| файл | rows | min | max | avg |
|------|------|-----|-----|-----|
| prompts.csv | 500 | 259 | 877 | 532 |
| responses.csv | 500 | 261 | 894 | 529 |

## Litserve Baseline


