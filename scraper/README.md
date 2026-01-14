# Сбор данных для CS2 Predictor

## 🚀 БЫСТРЫЙ СТАРТ (5 минут)
KGAT_a626a3496d2fe124f382a45f41e24a8a
### Шаг 1: Настройка Kaggle API

```bash
# 1. Зайдите на https://www.kaggle.com/settings
# 2. Прокрутите до секции "API"
# 3. Нажмите "Create New Token" - скачается kaggle.json
# 4. Переместите файл:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Шаг 2: Скачивание датасета

```bash
cd /home/vladbily/PycharmProjects/cs-predictor

# Установка
pip install kaggle

# Скачивание (выберите один датасет)
# Вариант 1: 50K+ профессиональных матчей
kaggle datasets download -d mateusdmachado/csgo-professional-matches -p data/kaggle

# Вариант 2: HLTV данные
kaggle datasets download -d danielgordon/csgo-professional-match-data -p data/kaggle

# Распаковка
cd data/kaggle && unzip -o *.zip && cd ../..
```

### Шаг 3: Обработка данных

```bash
# Конвертация в наш формат
python scraper/convert_kaggle_data.py

# Расчёт фичей
python data_processor.py

# Обучение модели
python train.py
```

---

## 📊 Альтернативные источники

### HLTV (ручной парсинг)

⚠️ **HLTV активно блокирует скраперы (403 Forbidden)**

Для успешного парсинга нужно:
1. Использовать **прокси** (rotating proxies)
2. Добавить **большие задержки** (10-30 сек)
3. Имитировать **реальный браузер** (Selenium/Playwright)

```bash
# Установка для Selenium
pip install selenium webdriver-manager

# Или для Playwright  
pip install playwright
playwright install
```

### PandaScore API (платный)

```bash
# Регистрация: https://pandascore.co/
# Бесплатно: 1000 запросов/день
```

```python
import requests
API_KEY = "your_key"
url = f"https://api.pandascore.co/csgo/matches?token={API_KEY}"
matches = requests.get(url).json()
```

---

## 📁 Структура датасета

Минимально необходимые колонки:
```
date, team_A, team_B, map, winner
```

Расширенный набор (для лучшего accuracy):
```
date, team_A, team_B, map, winner
team_A_rank, team_B_rank
event_name, event_type (Major/Online/LAN)
picked_by_is_A, is_decider
team_A_avg_rating, team_B_avg_rating
team_A_avg_kd, team_B_avg_kd
team_A_avg_adr, team_B_avg_adr
has_standin_A, has_standin_B
```

---

## ⏱️ Время сбора данных

| Метод | 50K матчей | Complexity |
|-------|-----------|-----------|
| **Kaggle** | **5 минут** | ⭐ Легко |
| Свой скрапер + прокси | 5-10 дней | ⭐⭐⭐ Сложно |
| PandaScore API | 50+ дней | ⭐⭐ Средне |

---

## 📂 Файлы в этой папке

```
scraper/
├── README.md                  # Эта инструкция
├── download_kaggle.sh         # Скрипт скачивания с Kaggle
├── convert_kaggle_data.py     # Конвертация Kaggle данных
├── hltv_scraper.py           # Парсер HLTV (нужны прокси!)
├── hltv_api_collector.py     # Сборщик через API
├── process_hltv_data.py      # Обработка данных HLTV
└── alternative_sources.py    # Другие источники
```
