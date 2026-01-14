"""
HLTV Scraper с использованием Selenium для обхода блокировок.
Selenium имитирует реальный браузер, что позволяет обойти защиту от ботов.

Требования:
    pip install selenium webdriver-manager pandas
"""

import pandas as pd
import time
import random
import json
import os
from datetime import datetime
import logging

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium не установлен. Установите: pip install selenium webdriver-manager")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HLTVSeleniumScraper:
    """Скрапер HLTV с использованием Selenium для обхода защиты"""
    
    BASE_URL = "https://www.hltv.org"
    
    def __init__(self, headless=True):
        """
        Args:
            headless: запускать браузер без GUI (быстрее)
        """
        if not SELENIUM_AVAILABLE:
            raise ImportError("Установите selenium: pip install selenium webdriver-manager")
        
        self.options = Options()
        if headless:
            self.options.add_argument("--headless=new")
        
        # Настройки для обхода детектирования
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        self.driver = None
        
    def start(self):
        """Запустить браузер"""
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=self.options)
        
        # Скрываем признаки автоматизации
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        logger.info("Browser started")
        
    def close(self):
        """Закрыть браузер"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")
    
    def random_delay(self, min_sec=2, max_sec=5):
        """Случайная задержка"""
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)
    
    def get_results_page(self, offset=0):
        """Получить страницу результатов"""
        url = f"{self.BASE_URL}/results?offset={offset}"
        self.driver.get(url)
        self.random_delay(3, 6)
        
        matches = []
        try:
            # Ждём загрузки результатов
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "result-con"))
            )
            
            result_divs = self.driver.find_elements(By.CLASS_NAME, "result-con")
            
            for div in result_divs:
                try:
                    link = div.find_element(By.CSS_SELECTOR, "a.a-reset")
                    href = link.get_attribute("href")
                    match_id = href.split("/")[4] if href else None
                    
                    teams = div.find_elements(By.CLASS_NAME, "team")
                    team1 = teams[0].text if len(teams) > 0 else ""
                    team2 = teams[1].text if len(teams) > 1 else ""
                    
                    scores = div.find_elements(By.CLASS_NAME, "result-score")
                    score1 = scores[0].text if len(scores) > 0 else ""
                    score2 = scores[1].text if len(scores) > 1 else ""
                    
                    event = div.find_element(By.CLASS_NAME, "event-name")
                    event_name = event.text if event else ""
                    
                    matches.append({
                        'match_id': match_id,
                        'url': href,
                        'team1': team1,
                        'team2': team2,
                        'score1': score1,
                        'score2': score2,
                        'event': event_name
                    })
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.error(f"Error getting results: {e}")
            
        return matches
    
    def get_match_details(self, match_url):
        """Получить детали матча"""
        self.driver.get(match_url)
        self.random_delay(3, 7)
        
        match_data = {'url': match_url}
        
        try:
            # Дата
            try:
                date_div = self.driver.find_element(By.CLASS_NAME, "date")
                match_data['date'] = date_div.get_attribute("data-unix")
            except:
                pass
            
            # Команды
            try:
                teams = self.driver.find_elements(By.CLASS_NAME, "teamName")
                if len(teams) >= 2:
                    match_data['team1'] = teams[0].text
                    match_data['team2'] = teams[1].text
            except:
                pass
            
            # LAN/Online
            try:
                self.driver.find_element(By.CLASS_NAME, "lan-icon")
                match_data['is_lan'] = True
            except:
                match_data['is_lan'] = False
            
            # Турнир
            try:
                event = self.driver.find_element(By.CSS_SELECTOR, ".event a")
                match_data['event'] = event.text
            except:
                pass
            
            # Карты
            maps = []
            try:
                map_holders = self.driver.find_elements(By.CLASS_NAME, "mapholder")
                for mh in map_holders:
                    try:
                        map_name = mh.find_element(By.CLASS_NAME, "mapname").text
                        results = mh.find_elements(By.CLASS_NAME, "results-team-score")
                        
                        map_data = {
                            'map': map_name,
                            'score1': results[0].text if len(results) > 0 else '',
                            'score2': results[1].text if len(results) > 1 else ''
                        }
                        
                        try:
                            picked = mh.find_element(By.CLASS_NAME, "pick")
                            map_data['picked_by'] = picked.text
                        except:
                            map_data['picked_by'] = ''
                        
                        maps.append(map_data)
                    except:
                        continue
            except:
                pass
            
            match_data['maps'] = maps
            
            # Статистика игроков
            players = []
            try:
                # Переходим на вкладку статистики если есть
                stats_tables = self.driver.find_elements(By.CLASS_NAME, "stats-table")
                
                for table in stats_tables:
                    rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
                    for row in rows:
                        try:
                            player_link = row.find_element(By.CSS_SELECTOR, ".st-player a")
                            player_name = player_link.text
                            player_id = player_link.get_attribute("href").split("/")[4]
                            
                            cells = row.find_elements(By.TAG_NAME, "td")
                            
                            player_data = {
                                'name': player_name,
                                'id': player_id,
                                'kills': cells[1].text if len(cells) > 1 else '',
                                'deaths': cells[3].text if len(cells) > 3 else '',
                                'rating': cells[-1].text if cells else ''
                            }
                            players.append(player_data)
                        except:
                            continue
            except:
                pass
            
            match_data['players'] = players
            
        except Exception as e:
            logger.error(f"Error parsing match: {e}")
            
        return match_data
    
    def get_team_rankings(self):
        """Получить рейтинг команд"""
        url = f"{self.BASE_URL}/ranking/teams"
        self.driver.get(url)
        self.random_delay(3, 6)
        
        rankings = []
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "ranked-team"))
            )
            
            team_rows = self.driver.find_elements(By.CLASS_NAME, "ranked-team")
            
            for row in team_rows:
                try:
                    rank = row.find_element(By.CLASS_NAME, "position").text.replace("#", "")
                    name = row.find_element(By.CLASS_NAME, "name").text
                    points = row.find_element(By.CLASS_NAME, "points").text
                    
                    rankings.append({
                        'rank': rank,
                        'team': name,
                        'points': points
                    })
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Error getting rankings: {e}")
            
        return rankings


def scrape_with_selenium(n_matches=1000, output_dir='data/selenium_scraped'):
    """
    Собрать матчи с помощью Selenium.
    
    Args:
        n_matches: количество матчей
        output_dir: папка для сохранения
    """
    if not SELENIUM_AVAILABLE:
        print("Установите selenium: pip install selenium webdriver-manager")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    scraper = HLTVSeleniumScraper(headless=True)
    
    try:
        scraper.start()
        
        all_matches = []
        all_maps = []
        
        # Загружаем чекпоинт
        checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                all_matches = checkpoint.get('matches', [])
                logger.info(f"Loaded {len(all_matches)} matches from checkpoint")
        
        collected_ids = {m['match_id'] for m in all_matches if m.get('match_id')}
        offset = (len(collected_ids) // 50) * 50
        
        while len(all_matches) < n_matches:
            logger.info(f"Fetching results page, offset={offset}, collected={len(all_matches)}")
            
            results = scraper.get_results_page(offset=offset)
            
            if not results:
                logger.warning("No results, waiting...")
                time.sleep(30)
                continue
            
            for match_info in results:
                if match_info['match_id'] in collected_ids:
                    continue
                    
                if len(all_matches) >= n_matches:
                    break
                
                logger.info(f"Fetching match: {match_info['team1']} vs {match_info['team2']}")
                
                details = scraper.get_match_details(match_info['url'])
                
                if details:
                    match_data = {**match_info, **details}
                    all_matches.append(match_data)
                    collected_ids.add(match_info['match_id'])
                    
                    # Добавляем карты
                    for map_data in details.get('maps', []):
                        all_maps.append({
                            'match_id': match_info['match_id'],
                            **match_info,
                            **map_data
                        })
                    
                    # Чекпоинт каждые 50 матчей
                    if len(all_matches) % 50 == 0:
                        with open(checkpoint_file, 'w') as f:
                            json.dump({'matches': all_matches}, f, default=str)
                        logger.info(f"Checkpoint: {len(all_matches)} matches")
            
            offset += 50
        
        # Финальное сохранение
        df_matches = pd.DataFrame(all_matches)
        df_matches.to_csv(os.path.join(output_dir, 'matches.csv'), index=False)
        
        df_maps = pd.DataFrame(all_maps)
        df_maps.to_csv(os.path.join(output_dir, 'maps.csv'), index=False)
        
        logger.info(f"Done! Saved {len(all_matches)} matches, {len(all_maps)} maps")
        
    finally:
        scraper.close()


if __name__ == "__main__":
    print("="*60)
    print("HLTV Selenium Scraper")
    print("="*60)
    print()
    print("Этот скрапер использует Selenium для обхода защиты HLTV.")
    print("Для 1000 матчей потребуется ~3-5 часов.")
    print()
    
    # Тест на небольшом количестве
    scrape_with_selenium(n_matches=10, output_dir='data/selenium_test')
