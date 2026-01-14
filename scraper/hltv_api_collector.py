"""
Быстрый сбор данных через hltv-async-api
Это РЕКОМЕНДУЕМЫЙ способ - быстрее и надёжнее чем свой скрапер
"""

import asyncio
import pandas as pd
import json
import os
from datetime import datetime
from hltv_async_api import Hltv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def collect_matches(n_pages=100, output_dir='data/hltv_api'):
    """
    Собрать матчи через HLTV API.
    
    Args:
        n_pages: количество страниц результатов (каждая ~50 матчей)
        output_dir: папка для сохранения
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_matches = []
    all_maps = []
    all_player_stats = []
    
    # Загружаем чекпоинт если есть
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
    start_page = 1
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            all_matches = checkpoint.get('matches', [])
            start_page = checkpoint.get('last_page', 1) + 1
            logger.info(f"Resuming from page {start_page}, {len(all_matches)} matches collected")
    
    async with Hltv() as hltv:
        for page in range(start_page, n_pages + 1):
            try:
                logger.info(f"Fetching page {page}/{n_pages}...")
                
                # Получаем список результатов
                results = await hltv.get_results(offset=(page-1)*50)
                
                if not results:
                    logger.warning(f"No results on page {page}")
                    continue
                
                for match in results:
                    try:
                        match_id = match.get('id')
                        if not match_id:
                            continue
                        
                        # Пропускаем если уже есть
                        if any(m.get('match_id') == match_id for m in all_matches):
                            continue
                        
                        logger.info(f"  Fetching match {match_id}: {match.get('team1', {}).get('name')} vs {match.get('team2', {}).get('name')}")
                        
                        # Получаем детали матча
                        details = await hltv.get_match(match_id)
                        
                        if details:
                            match_data = {
                                'match_id': match_id,
                                'date': details.get('date'),
                                'team1': details.get('team1', {}).get('name'),
                                'team1_id': details.get('team1', {}).get('id'),
                                'team2': details.get('team2', {}).get('name'),
                                'team2_id': details.get('team2', {}).get('id'),
                                'event': details.get('event', {}).get('name'),
                                'event_id': details.get('event', {}).get('id'),
                                'winner': 'team1' if details.get('winner') == details.get('team1', {}).get('name') else 'team2'
                            }
                            
                            # Добавляем карты
                            for map_data in details.get('maps', []):
                                map_record = {
                                    'match_id': match_id,
                                    'map': map_data.get('name'),
                                    'result_team1': map_data.get('result', {}).get('team1'),
                                    'result_team2': map_data.get('result', {}).get('team2'),
                                    **match_data
                                }
                                all_maps.append(map_record)
                            
                            # Добавляем статистику игроков
                            for player in details.get('players', []):
                                player_record = {
                                    'match_id': match_id,
                                    'player_id': player.get('id'),
                                    'player_name': player.get('name'),
                                    'team': player.get('team'),
                                    'kills': player.get('kills'),
                                    'deaths': player.get('deaths'),
                                    'adr': player.get('adr'),
                                    'kast': player.get('kast'),
                                    'rating': player.get('rating')
                                }
                                all_player_stats.append(player_record)
                            
                            all_matches.append(match_data)
                        
                        # Задержка между запросами
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error processing match: {e}")
                        continue
                
                # Сохраняем чекпоинт каждые 5 страниц
                if page % 5 == 0:
                    save_checkpoint(all_matches, page, output_dir)
                    logger.info(f"Checkpoint saved: {len(all_matches)} matches")
                
                # Задержка между страницами
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Error on page {page}: {e}")
                await asyncio.sleep(30)
                continue
    
    # Финальное сохранение
    save_all_data(all_matches, all_maps, all_player_stats, output_dir)
    
    return len(all_matches)


def save_checkpoint(matches, page, output_dir):
    """Сохранить промежуточный чекпоинт"""
    checkpoint = {
        'matches': matches,
        'last_page': page,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, 'checkpoint.json'), 'w') as f:
        json.dump(checkpoint, f, default=str)


def save_all_data(matches, maps, player_stats, output_dir):
    """Сохранить все данные"""
    # Матчи
    df_matches = pd.DataFrame(matches)
    df_matches.to_csv(os.path.join(output_dir, 'matches.csv'), index=False)
    
    # Карты
    df_maps = pd.DataFrame(maps)
    df_maps.to_csv(os.path.join(output_dir, 'maps.csv'), index=False)
    
    # Статистика игроков
    df_players = pd.DataFrame(player_stats)
    df_players.to_csv(os.path.join(output_dir, 'player_stats.csv'), index=False)
    
    logger.info(f"Saved: {len(df_matches)} matches, {len(df_maps)} maps, {len(df_players)} player stats")


async def get_team_rankings():
    """Получить текущий рейтинг команд"""
    async with Hltv() as hltv:
        rankings = await hltv.get_team_ranking()
        return rankings


async def get_player_info(player_id):
    """Получить информацию об игроке"""
    async with Hltv() as hltv:
        player = await hltv.get_player(player_id)
        return player


if __name__ == "__main__":
    # Пример: собрать 10 страниц (~500 матчей) для теста
    print("Starting data collection...")
    print("This may take a while due to rate limiting...")
    
    n_collected = asyncio.run(collect_matches(n_pages=10, output_dir='data/hltv_api'))
    
    print(f"\nDone! Collected {n_collected} matches")
    print("Files saved in data/hltv_api/")
