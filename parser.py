import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from loguru import logger


class CS2FullDataScraper:
    def __init__(self):
        self.base_url = "https://www.hltv.org"
        self.results_url = "https://www.hltv.org/results"

    async def fetch_with_retry(self, page, url, selector, retries=3):
        for i in range(retries):
            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)
                await page.wait_for_selector(selector, timeout=10000)
                return True
            except Exception as e:
                logger.warning(f"Попытка {i + 1} не удалась для {url}: {e}")
                await asyncio.sleep(5)
        return False

    async def parse_player_stats(self, page, match_url):
        stats_url = match_url.replace("/matches/", "/stats/matches/")
        await page.goto(stats_url, wait_until="networkidle")

        try:
            await page.wait_for_selector(".stats-table")

            rows = await page.query_selector_all(".stats-table tbody tr")

            all_players = []
            for row in rows:
                cols = await row.query_selector_all("td")
                if len(cols) < 5: continue

                rating = await cols[5].inner_text()
                adr = await cols[3].inner_text()

                all_players.append({
                    'rating': float(rating),
                    'adr': float(adr)
                })

            if len(all_players) < 10: return None

            team_a = all_players[:5]
            team_b = all_players[5:]

            stats = {
                'avg_rating_A': sum(p['rating'] for p in team_a) / 5,
                'avg_rating_B': sum(p['rating'] for p in team_b) / 5,
                'max_rating_A': max(p['rating'] for p in team_a),
                'max_rating_B': max(p['rating'] for p in team_b),
                'total_adr_A': sum(p['adr'] for p in team_a),
                'total_adr_B': sum(p['adr'] for p in team_b)
            }
            return stats
        except Exception as e:
            logger.error(f"Ошибка парсинга игроков: {e}")
            return None

    async def parse_match(self, page, url):
        if not await self.fetch_with_retry(page, url, ".match-page"):
            return []

        try:
            match_id = url.split("/")[-2]
            team_a = await page.locator(".team1 .teamName").inner_text()
            team_b = await page.locator(".team2 .teamName").inner_text()

            rank_a_raw = await page.locator(".team1 .teamRanking").inner_text()
            rank_b_raw = await page.locator(".team2 .teamRanking").inner_text()
            rank_a = int(''.join(filter(str.isdigit, rank_a_raw))) if any(c.isdigit() for c in rank_a_raw) else 100
            rank_b = int(''.join(filter(str.isdigit, rank_b_raw))) if any(c.isdigit() for c in rank_b_raw) else 100

            map_holders = await page.query_selector_all(".mapholder")
            match_results = []

            player_stats = await self.parse_player_stats(page, url)
            if not player_stats: return []

            for holder in map_holders:
                map_name = await (await holder.query_selector(".mapname")).inner_text()
                if "Default" in map_name: continue

                pistols_a, pistols_b = 0, 0
                round_history = await holder.query_selector_all(".round-history-outcome")
                if len(round_history) >= 13:
                    for r_idx in [0, 12]:
                        img_src = await round_history[r_idx].get_attribute("src")
                        if "team1" in img_src:
                            pistols_a += 1
                        else:
                            pistols_b += 1

                score_el = await holder.query_selector(".results")
                score_text = await score_el.inner_text() if score_el else "0:0"
                s_a, s_b = map(int, score_text.replace("(", "").replace(")", "").split(":"))

                res = {
                    "match_id": match_id,
                    "map": map_name,
                    "team_A": team_a,
                    "team_B": team_b,
                    "team_A_rank": rank_a,
                    "team_B_rank": rank_b,
                    "pistol_wins_A": pistols_a,
                    "pistol_wins_B": pistols_b,
                    "winner_is_A": 1 if s_a > s_b else 0
                }
                res.update(player_stats)
                match_results.append(res)

            return match_results
        except Exception as e:
            logger.error(f"Ошибка в матче {url}: {e}")
            return []

    async def run(self, num_matches=20):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto(self.results_url, wait_until="networkidle")
            links = await page.eval_on_selector_all(".a-reset",
                                                    "els => els.map(e => e.href).filter(h => h.includes('/matches/'))")
            unique_links = list(dict.fromkeys(links))[:num_matches]

            final_data = []
            for link in unique_links:
                logger.info(f"Парсинг: {link}")
                data = await self.parse_match(page, link)
                final_data.extend(data)
                await asyncio.sleep(3)

            await browser.close()
            return pd.DataFrame(final_data)


if __name__ == "__main__":
    scraper = CS2FullDataScraper()
    df = asyncio.run(scraper.run(num_matches=10))
    df.to_csv("cs2_mega_dataset.csv", index=False)
    print("Данные собраны!")