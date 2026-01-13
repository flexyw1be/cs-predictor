import asyncio
import pandas as pd
from playwright.async_api import async_playwright
# Вместо import playwright_stealth
# или вместо from playwright_stealth import stealth (если это не сработало)
# Сделай так:
from playwright_stealth import stealth
from bs4 import BeautifulSoup


class HLTVParser:
    def __init__(self):
        self.results_url = "https://www.hltv.org/results"
        self.base_url = "https://www.hltv.org"

    async def run(self, num_matches=20):
        async with async_playwright() as p:
            # Запуск браузера с настройками против детекции ботов
            browser = await p.chromium.launch(headless=True)  # Поставь False, чтобы видеть процесс
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080}
            )

            page = await context.new_page()
            # Применяем стелс-режим
            await stealth(page)

            print(f"Переходим на {self.results_url}...")

            try:
                # Ждем загрузки только основного контента, чтобы не виснуть на рекламе
                await page.goto(self.results_url, wait_until="domcontentloaded", timeout=60000)
                # Ждем появления контейнера с результатами
                await page.wait_for_selector(".results-all", timeout=10000)

                html = await page.content()
                matches = self.parse_results_page(html, num_matches)

                df = pd.DataFrame(matches)
                df.to_csv("cs_matches.csv", index=False)
                print(f"Готово! Сохранено {len(df)} матчей в cs_matches.csv")
                return df

            except Exception as e:
                print(f"Произошла ошибка: {e}")
            finally:
                await browser.close()

    def parse_results_page(self, html, limit):
        soup = BeautifulSoup(html, "html.parser")
        match_list = []

        # Находим блоки результатов
        result_conts = soup.select(".result-con")

        for match in result_conts[:limit]:
            try:
                # Извлекаем данные
                team1 = match.select_one(".team1 .team").text.strip()
                team2 = match.select_one(".team2 .team").text.strip()

                # Счет (например "16 - 12")
                score_t1 = match.select_one(".score-won, .score-lost").text.strip()
                score_t2 = match.select(".score-won, .score-lost")[1].text.strip()

                event = match.select_one(".event-name").text.strip()
                map_type = match.select_one(".map-text").text.strip()  # "bo1", "bo3" и т.д.

                # Победитель для обучения модели (Target)
                winner = 1 if int(score_t1) > int(score_t2) else 0

                match_list.append({
                    "team1": team1,
                    "team2": team2,
                    "score_t1": int(score_t1),
                    "score_t2": int(score_t2),
                    "event": event,
                    "map_type": map_type,
                    "winner": winner
                })
            except (AttributeError, ValueError, IndexError):
                continue

        return match_list


if __name__ == "__main__":
    scraper = HLTVParser()
    asyncio.run(scraper.run(num_matches=50))