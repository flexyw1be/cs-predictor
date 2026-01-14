"""
HLTV.org Scraper для сбора данных о матчах CS2/CS:GO

ВАЖНО: 
- HLTV блокирует частые запросы, используйте задержки
- Рекомендуется использовать прокси для массового парсинга
- Соблюдайте robots.txt и Terms of Service
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HLTVScraper:
    BASE_URL = "https://www.hltv.org"
    
    # Headers для имитации браузера
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    def __init__(self, delay_range=(3, 7), use_proxy=False, proxy_list=None):
        """
        Args:
            delay_range: (min, max) секунд между запросами
            use_proxy: использовать прокси
            proxy_list: список прокси в формате ['http://ip:port', ...]
        """
        self.delay_range = delay_range
        self.use_proxy = use_proxy
        self.proxy_list = proxy_list or []
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        
    def _get_proxy(self) -> Optional[Dict]:
        """Получить случайный прокси"""
        if self.use_proxy and self.proxy_list:
            proxy = random.choice(self.proxy_list)
            return {'http': proxy, 'https': proxy}
        return None
    
    def _request(self, url: str, max_retries=3) -> Optional[BeautifulSoup]:
        """Выполнить запрос с повторами и задержкой"""
        for attempt in range(max_retries):
            try:
                # Случайная задержка
                delay = random.uniform(*self.delay_range)
                time.sleep(delay)
                
                proxies = self._get_proxy()
                response = self.session.get(url, proxies=proxies, timeout=30)
                
                if response.status_code == 200:
                    return BeautifulSoup(response.text, 'html.parser')
                elif response.status_code == 403:
                    logger.warning(f"Blocked by HLTV (403). Waiting 60s...")
                    time.sleep(60)
                elif response.status_code == 429:
                    logger.warning(f"Rate limited (429). Waiting 120s...")
                    time.sleep(120)
                else:
                    logger.warning(f"Status {response.status_code} for {url}")
                    
            except Exception as e:
                logger.error(f"Request error: {e}")
                time.sleep(10)
                
        return None
    
    def get_match_list(self, offset=0, stars=1) -> List[Dict]:
        """
        Получить список матчей.
        
        Args:
            offset: смещение для пагинации
            stars: минимальный рейтинг матча (1-5)
        """
        url = f"{self.BASE_URL}/results?offset={offset}&stars={stars}"
        soup = self._request(url)
        
        if not soup:
            return []
        
        matches = []
        result_divs = soup.select('.result-con')
        
        for div in result_divs:
            try:
                match_link = div.select_one('a.a-reset')
                if not match_link:
                    continue
                    
                match_id = match_link['href'].split('/')[2]
                
                teams = div.select('.team')
                if len(teams) < 2:
                    continue
                    
                team1 = teams[0].text.strip()
                team2 = teams[1].text.strip()
                
                scores = div.select('.result-score')
                score1 = scores[0].text.strip() if scores else '0'
                score2 = scores[1].text.strip() if len(scores) > 1 else '0'
                
                event = div.select_one('.event-name')
                event_name = event.text.strip() if event else ''
                
                matches.append({
                    'match_id': match_id,
                    'team1': team1,
                    'team2': team2,
                    'score1': score1,
                    'score2': score2,
                    'event': event_name,
                    'url': self.BASE_URL + match_link['href']
                })
                
            except Exception as e:
                logger.error(f"Error parsing match: {e}")
                continue
                
        return matches
    
    def get_match_details(self, match_id: str) -> Optional[Dict]:
        """Получить детали матча включая статистику игроков"""
        url = f"{self.BASE_URL}/matches/{match_id}/match"
        soup = self._request(url)
        
        if not soup:
            return None
            
        try:
            match_data = {
                'match_id': match_id,
                'maps': [],
                'players_stats': []
            }
            
            # Дата матча
            date_div = soup.select_one('.date')
            if date_div:
                match_data['date'] = date_div.get('data-unix', '')
            
            # Команды
            teams = soup.select('.teamName')
            if len(teams) >= 2:
                match_data['team1'] = teams[0].text.strip()
                match_data['team2'] = teams[1].text.strip()
            
            # Результат
            scores = soup.select('.team .won, .team .lost')
            
            # Турнир
            event = soup.select_one('.event a')
            if event:
                match_data['event'] = event.text.strip()
            
            # Тип матча (LAN/Online)
            lan_icon = soup.select_one('.lan-icon')
            match_data['is_lan'] = lan_icon is not None
            
            # Карты
            maps_div = soup.select('.mapholder')
            for map_div in maps_div:
                map_name = map_div.select_one('.mapname')
                if map_name:
                    results = map_div.select('.results-team-score')
                    picked_by = map_div.select_one('.pick')
                    
                    map_data = {
                        'map': map_name.text.strip(),
                        'score1': results[0].text.strip() if results else '',
                        'score2': results[1].text.strip() if len(results) > 1 else '',
                        'picked_by': picked_by.text.strip() if picked_by else ''
                    }
                    match_data['maps'].append(map_data)
            
            # Статистика игроков (из страницы матча)
            stats_tables = soup.select('.stats-table')
            for table in stats_tables:
                rows = table.select('tbody tr')
                for row in rows:
                    player_cell = row.select_one('.st-player a')
                    if not player_cell:
                        continue
                        
                    player_name = player_cell.text.strip()
                    player_id = player_cell['href'].split('/')[2] if player_cell.get('href') else ''
                    
                    cells = row.select('td.st-kills, td.st-assists, td.st-deaths, td.st-kdratio, td.st-adr, td.st-rating')
                    
                    stats = {
                        'player_name': player_name,
                        'player_id': player_id,
                        'kills': cells[0].text.strip() if len(cells) > 0 else '',
                        'assists': cells[1].text.strip() if len(cells) > 1 else '',
                        'deaths': cells[2].text.strip() if len(cells) > 2 else '',
                        'kd': cells[3].text.strip() if len(cells) > 3 else '',
                        'adr': cells[4].text.strip() if len(cells) > 4 else '',
                        'rating': cells[5].text.strip() if len(cells) > 5 else ''
                    }
                    match_data['players_stats'].append(stats)
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error parsing match details: {e}")
            return None
    
    def get_player_stats(self, player_id: str, time_filter='3m') -> Optional[Dict]:
        """
        Получить статистику игрока за период.
        
        Args:
            player_id: ID игрока на HLTV
            time_filter: период ('1m', '3m', '6m', '12m', 'all')
        """
        url = f"{self.BASE_URL}/stats/players/{player_id}?startDate=all&endDate=all&matchType=all"
        soup = self._request(url)
        
        if not soup:
            return None
            
        try:
            stats = {'player_id': player_id}
            
            # Имя игрока
            name = soup.select_one('.summaryNickname')
            if name:
                stats['name'] = name.text.strip()
            
            # Команда
            team = soup.select_one('.SummaryTeamname')
            if team:
                stats['team'] = team.text.strip()
            
            # Основные статистики
            stat_rows = soup.select('.summaryStatBreakdownRow')
            for row in stat_rows:
                label = row.select_one('.summaryStatBreakdownSubHeader')
                value = row.select_one('.summaryStatBreakdownDataValue')
                if label and value:
                    key = label.text.strip().lower().replace(' ', '_')
                    stats[key] = value.text.strip()
            
            # Детальные статистики
            detail_stats = soup.select('.statistics .columns .col .stats-row')
            for stat in detail_stats:
                spans = stat.select('span')
                if len(spans) >= 2:
                    key = spans[0].text.strip().lower().replace(' ', '_').replace('/', '_')
                    stats[key] = spans[1].text.strip()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing player stats: {e}")
            return None
    
    def get_team_info(self, team_id: str) -> Optional[Dict]:
        """Получить информацию о команде"""
        url = f"{self.BASE_URL}/team/{team_id}/team"
        soup = self._request(url)
        
        if not soup:
            return None
            
        try:
            team_data = {'team_id': team_id}
            
            # Название
            name = soup.select_one('.profile-team-name')
            if name:
                team_data['name'] = name.text.strip()
            
            # Рейтинг
            rank = soup.select_one('.profile-team-stat .right')
            if rank:
                team_data['world_rank'] = rank.text.strip().replace('#', '')
            
            # Игроки
            players = []
            player_divs = soup.select('.bodyshot-team-bg a')
            for p in player_divs:
                player_name = p.select_one('.text-ellipsis')
                if player_name:
                    players.append({
                        'name': player_name.text.strip(),
                        'id': p['href'].split('/')[2] if p.get('href') else ''
                    })
            team_data['players'] = players
            
            return team_data
            
        except Exception as e:
            logger.error(f"Error parsing team info: {e}")
            return None

    def get_rankings(self) -> List[Dict]:
        """Получить текущий рейтинг команд"""
        url = f"{self.BASE_URL}/ranking/teams"
        soup = self._request(url)
        
        if not soup:
            return []
            
        rankings = []
        try:
            team_rows = soup.select('.ranked-team')
            for row in team_rows:
                rank = row.select_one('.position')
                name = row.select_one('.name')
                points = row.select_one('.points')
                
                if rank and name:
                    rankings.append({
                        'rank': rank.text.strip().replace('#', ''),
                        'team': name.text.strip(),
                        'points': points.text.strip() if points else ''
                    })
                    
        except Exception as e:
            logger.error(f"Error parsing rankings: {e}")
            
        return rankings


def scrape_matches(n_matches=50000, output_dir='data/scraped', checkpoint_every=100):
    """
    Главная функция для сбора матчей.
    
    Args:
        n_matches: количество матчей для сбора
        output_dir: папка для сохранения
        checkpoint_every: сохранять каждые N матчей
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scraper = HLTVScraper(delay_range=(5, 10))
    
    all_matches = []
    all_maps = []
    all_player_stats = []
    
    # Загрузка чекпоинта если есть
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            all_matches = checkpoint.get('matches', [])
            logger.info(f"Loaded {len(all_matches)} matches from checkpoint")
    
    collected_ids = {m['match_id'] for m in all_matches}
    offset = len(collected_ids) // 100 * 100
    
    while len(all_matches) < n_matches:
        logger.info(f"Fetching match list, offset={offset}, collected={len(all_matches)}")
        
        # Получаем список матчей
        match_list = scraper.get_match_list(offset=offset, stars=1)
        
        if not match_list:
            logger.warning("No matches found, waiting and retrying...")
            time.sleep(60)
            continue
        
        for match_info in match_list:
            if match_info['match_id'] in collected_ids:
                continue
                
            if len(all_matches) >= n_matches:
                break
            
            logger.info(f"Fetching match {match_info['match_id']}: {match_info['team1']} vs {match_info['team2']}")
            
            # Получаем детали матча
            details = scraper.get_match_details(match_info['match_id'])
            
            if details:
                # Объединяем данные
                match_data = {**match_info, **details}
                all_matches.append(match_data)
                collected_ids.add(match_info['match_id'])
                
                # Добавляем карты
                for map_data in details.get('maps', []):
                    map_record = {
                        'match_id': match_info['match_id'],
                        **match_info,
                        **map_data
                    }
                    all_maps.append(map_record)
                
                # Добавляем статистику игроков
                for player_stat in details.get('players_stats', []):
                    stat_record = {
                        'match_id': match_info['match_id'],
                        **player_stat
                    }
                    all_player_stats.append(stat_record)
                
                # Сохраняем чекпоинт
                if len(all_matches) % checkpoint_every == 0:
                    save_checkpoint(all_matches, all_maps, all_player_stats, output_dir)
                    logger.info(f"Checkpoint saved: {len(all_matches)} matches")
        
        offset += 100
    
    # Финальное сохранение
    save_data(all_matches, all_maps, all_player_stats, output_dir)
    logger.info(f"Done! Collected {len(all_matches)} matches, {len(all_maps)} maps")


def save_checkpoint(matches, maps, player_stats, output_dir):
    """Сохранить промежуточный чекпоинт"""
    checkpoint = {
        'matches': matches,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, 'checkpoint.json'), 'w') as f:
        json.dump(checkpoint, f)


def save_data(matches, maps, player_stats, output_dir):
    """Сохранить финальные данные"""
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


if __name__ == "__main__":
    # Пример: собрать 100 матчей для теста
    scrape_matches(n_matches=100, output_dir='data/scraped', checkpoint_every=10)
