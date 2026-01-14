"""
Альтернативные источники данных для CS2/CS:GO
"""

# ============================================================
# ВАРИАНТ 1: ГОТОВЫЕ ДАТАСЕТЫ (самый быстрый способ)
# ============================================================

KAGGLE_DATASETS = """
1. CS:GO Professional Matches (2015-2020)
   https://www.kaggle.com/datasets/mateusdmachado/csgo-professional-matches
   - 50,000+ матчей
   - Статистика игроков

2. HLTV CS:GO Statistics
   https://www.kaggle.com/datasets/danielgordon/csgo-professional-match-data
   
3. CS:GO Players Statistics
   https://www.kaggle.com/datasets/jtrotman/csgo-player-statistics

Скачать через Kaggle API:
    pip install kaggle
    kaggle datasets download -d mateusdmachado/csgo-professional-matches
"""

# ============================================================
# ВАРИАНТ 2: HLTV API (неофициальный)
# ============================================================

def example_hltv_api():
    """
    Использование пакета hltv-async-api
    pip install hltv-async-api
    """
    code = '''
import asyncio
from hltv_async_api import Hltv

async def get_matches():
    async with Hltv() as hltv:
        # Получить последние матчи
        matches = await hltv.get_results(pages=10)
        
        # Получить детали матча
        match_details = await hltv.get_match(match_id=12345)
        
        # Получить статистику игрока
        player = await hltv.get_player(player_id=7998)
        
        # Получить рейтинг команд
        rankings = await hltv.get_team_ranking()
        
        return matches

matches = asyncio.run(get_matches())
    '''
    return code


# ============================================================
# ВАРИАНТ 3: Liquipedia API
# ============================================================

def example_liquipedia_api():
    """
    Liquipedia имеет открытый API для получения данных о турнирах
    https://liquipedia.net/commons/Liquipedia:API_Usage_Guidelines
    """
    code = '''
import requests

def get_liquipedia_data(query):
    url = "https://liquipedia.net/counterstrike/api.php"
    params = {
        "action": "parse",
        "page": query,
        "format": "json"
    }
    headers = {
        "User-Agent": "YourBot/1.0 (your@email.com)"
    }
    response = requests.get(url, params=params, headers=headers)
    return response.json()

# Пример: получить информацию о турнире
data = get_liquipedia_data("IEM_Katowice_2024")
    '''
    return code


# ============================================================
# ВАРИАНТ 4: PandaScore API (коммерческий, но есть бесплатный tier)
# ============================================================

def example_pandascore_api():
    """
    PandaScore - профессиональный API для киберспорта
    https://pandascore.co/
    Бесплатный план: 1000 запросов/день
    """
    code = '''
import requests

API_KEY = "your_api_key"
BASE_URL = "https://api.pandascore.co"

def get_matches(page=1, per_page=100):
    url = f"{BASE_URL}/csgo/matches"
    params = {
        "token": API_KEY,
        "page": page,
        "per_page": per_page,
        "sort": "-begin_at"
    }
    response = requests.get(url, params=params)
    return response.json()

def get_player_stats(player_id):
    url = f"{BASE_URL}/csgo/players/{player_id}/stats"
    params = {"token": API_KEY}
    response = requests.get(url, params=params)
    return response.json()

# Собрать 50000 матчей
all_matches = []
for page in range(1, 501):
    matches = get_matches(page=page, per_page=100)
    all_matches.extend(matches)
    print(f"Page {page}: {len(all_matches)} matches")
    '''
    return code


# ============================================================
# ВАРИАНТ 5: Использовать готовые GitHub репозитории
# ============================================================

GITHUB_REPOS = """
Готовые скраперы и датасеты:

1. hltv-api (JavaScript)
   https://github.com/gigobyte/HLTV
   - Полный API для HLTV
   - npm install hltv

2. hltv-async-api (Python)
   https://github.com/akimerslern/hltv-async-api
   - pip install hltv-async-api

3. CSGOData
   https://github.com/Geertiansen/CSGOData
   - Готовые датасеты

4. esportsBettingModel
   https://github.com/josefkuzela/esportsBettingModel
   - Пример ML модели для CS:GO
"""


# ============================================================
# ВАРИАНТ 6: Быстрый старт - скачать готовый датасет
# ============================================================

def download_kaggle_dataset():
    """Скрипт для скачивания датасета с Kaggle"""
    import os
    import subprocess
    
    # Установка kaggle CLI
    subprocess.run(['pip', 'install', 'kaggle'])
    
    # Нужно настроить ~/.kaggle/kaggle.json с API ключом
    # Скачать можно с https://www.kaggle.com/settings
    
    # Скачивание датасета
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', 'mateusdmachado/csgo-professional-matches',
        '-p', 'data/kaggle'
    ])
    
    # Распаковка
    subprocess.run(['unzip', 'data/kaggle/*.zip', '-d', 'data/kaggle'])


if __name__ == "__main__":
    print("="*60)
    print("СПОСОБЫ ПОЛУЧЕНИЯ ДАННЫХ ДЛЯ CS2/CS:GO")
    print("="*60)
    print(KAGGLE_DATASETS)
    print("\nGitHub репозитории:")
    print(GITHUB_REPOS)
