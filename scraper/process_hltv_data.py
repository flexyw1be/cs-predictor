"""
Обработка собранных данных HLTV в формат для ML модели.
Объединяет матчи, карты и статистику игроков.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import os


def process_scraped_data(scraped_dir='data/scraped', output_dir='data/processed_hltv'):
    """Обработать сырые данные с HLTV"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Загрузка данных
    df_matches = pd.read_csv(os.path.join(scraped_dir, 'matches.csv'))
    df_maps = pd.read_csv(os.path.join(scraped_dir, 'maps.csv'))
    df_player_stats = pd.read_csv(os.path.join(scraped_dir, 'player_stats.csv'))
    
    print(f"Loaded: {len(df_matches)} matches, {len(df_maps)} maps, {len(df_player_stats)} player records")
    
    # === 1. АГРЕГАЦИЯ СТАТИСТИКИ ИГРОКОВ ПО КОМАНДАМ ===
    print("Calculating team stats from player data...")
    
    team_player_stats = calculate_team_player_stats(df_player_stats, df_matches)
    
    # === 2. ОБРАБОТКА КАРТочных ДАННЫХ ===
    print("Processing map data...")
    
    df_processed = process_maps(df_maps, df_matches, team_player_stats)
    
    # === 3. РАСЧЁТ ПРОДВИНУТЫХ ПРИЗНАКОВ ===
    print("Computing advanced features...")
    
    df_processed = compute_advanced_features_hltv(df_processed)
    
    # === 4. РАЗДЕЛЕНИЕ НА TRAIN/VAL/TEST ===
    print("Splitting data...")
    
    df_processed = df_processed.sort_values('date').reset_index(drop=True)
    
    n = len(df_processed)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    df_processed['split'] = 'train'
    df_processed.loc[train_end:val_end, 'split'] = 'val'
    df_processed.loc[val_end:, 'split'] = 'test'
    
    # Сохранение
    df_processed.to_csv(os.path.join(output_dir, 'full_dataset.csv'), index=False)
    df_processed[df_processed['split'] == 'train'].to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df_processed[df_processed['split'] == 'val'].to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    df_processed[df_processed['split'] == 'test'].to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Done! Saved {len(df_processed)} records")
    print(f"Train: {(df_processed['split']=='train').sum()}")
    print(f"Val: {(df_processed['split']=='val').sum()}")
    print(f"Test: {(df_processed['split']=='test').sum()}")
    
    return df_processed


def calculate_team_player_stats(df_player_stats, df_matches):
    """
    Рассчитать скользящую статистику команд на основе статистики игроков.
    Возвращает dict: (team, date) -> stats
    """
    
    # Объединяем с датами матчей
    df = df_player_stats.merge(df_matches[['match_id', 'date', 'team1', 'team2']], on='match_id')
    df['date'] = pd.to_datetime(df['date'])
    
    # Определяем команду игрока
    # (в реальности нужна более точная логика)
    
    # Группируем статистику по матчам и командам
    team_stats = defaultdict(lambda: deque(maxlen=20))  # последние 20 матчей
    
    # Результат
    result = {}
    
    for date in sorted(df['date'].unique()):
        matches_on_date = df[df['date'] == date]
        
        for match_id in matches_on_date['match_id'].unique():
            match_data = matches_on_date[matches_on_date['match_id'] == match_id]
            
            # Извлекаем статистику для каждой команды
            team1 = match_data['team1'].iloc[0]
            team2 = match_data['team2'].iloc[0]
            
            for team in [team1, team2]:
                # Сохраняем текущее состояние (до матча)
                if team_stats[team]:
                    recent = list(team_stats[team])
                    result[(team, date)] = {
                        'avg_rating': np.mean([s['rating'] for s in recent if s.get('rating')]),
                        'avg_kd': np.mean([s['kd'] for s in recent if s.get('kd')]),
                        'avg_adr': np.mean([s['adr'] for s in recent if s.get('adr')]),
                        'matches_played': len(recent)
                    }
                else:
                    result[(team, date)] = {
                        'avg_rating': 1.0,
                        'avg_kd': 1.0,
                        'avg_adr': 75.0,
                        'matches_played': 0
                    }
    
    return result


def process_maps(df_maps, df_matches, team_stats):
    """Обработать данные по картам"""
    
    df = df_maps.merge(
        df_matches[['match_id', 'date', 'event', 'is_lan']], 
        on='match_id', 
        how='left',
        suffixes=('', '_match')
    )
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Определяем победителя карты
    df['score1'] = pd.to_numeric(df['score1'], errors='coerce').fillna(0).astype(int)
    df['score2'] = pd.to_numeric(df['score2'], errors='coerce').fillna(0).astype(int)
    df['winner'] = np.where(df['score1'] > df['score2'], df['team1'], df['team2'])
    
    # Нормализуем названия команд
    df['team_A'] = df['team1']
    df['team_B'] = df['team2']
    df['winner_is_A'] = (df['winner'] == df['team_A']).astype(int)
    
    # Picked by
    df['picked_by_is_A'] = df.apply(
        lambda r: 1 if r.get('picked_by', '') == r['team_A'] else (0 if r.get('picked_by', '') == r['team_B'] else -1),
        axis=1
    )
    
    return df


def compute_advanced_features_hltv(df):
    """Вычислить все продвинутые признаки"""
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Инициализация хранилищ
    team_elo = defaultdict(lambda: 1500)
    team_map_elo = defaultdict(lambda: defaultdict(lambda: 1500))
    h2h_stats = defaultdict(lambda: {'wins_A': 0, 'total': 0})
    team_history = defaultdict(lambda: deque(maxlen=20))
    team_map_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=10)))
    team_streak = defaultdict(int)
    team_last_date = {}
    
    K_GLOBAL = 32
    K_MAP = 40
    
    # Списки для фичей
    features = {col: [] for col in [
        'elo_diff', 'map_elo_diff', 'h2h_rate', 'h2h_games',
        'momentum_diff', 'streak_A', 'streak_B',
        'days_since_last_A', 'days_since_last_B',
        'overall_winrate_A', 'overall_winrate_B',
    ]}
    
    for _, row in df.iterrows():
        a, b, m = row['team_A'], row['team_B'], row['map']
        current_date = row['date']
        
        # Elo
        features['elo_diff'].append(team_elo[a] - team_elo[b])
        features['map_elo_diff'].append(team_map_elo[a][m] - team_map_elo[b][m])
        
        # H2H
        h2h_key = tuple(sorted((a, b)))
        h2h = h2h_stats[h2h_key]
        if h2h['total'] > 0:
            rate = h2h['wins_A'] / h2h['total']
            h2h_rate = rate if a < b else 1 - rate
        else:
            h2h_rate = 0.5
        features['h2h_rate'].append(h2h_rate)
        features['h2h_games'].append(h2h['total'])
        
        # Momentum
        m_a = np.mean(team_map_history[a][m]) if team_map_history[a][m] else 0.5
        m_b = np.mean(team_map_history[b][m]) if team_map_history[b][m] else 0.5
        features['momentum_diff'].append(m_a - m_b)
        
        # Streak
        features['streak_A'].append(team_streak[a])
        features['streak_B'].append(team_streak[b])
        
        # Days since last
        if a in team_last_date:
            features['days_since_last_A'].append((current_date - team_last_date[a]).days)
        else:
            features['days_since_last_A'].append(30)
        if b in team_last_date:
            features['days_since_last_B'].append((current_date - team_last_date[b]).days)
        else:
            features['days_since_last_B'].append(30)
        
        # Winrates
        wr_a = np.mean([h[0] for h in team_history[a]]) if team_history[a] else 0.5
        wr_b = np.mean([h[0] for h in team_history[b]]) if team_history[b] else 0.5
        features['overall_winrate_A'].append(wr_a)
        features['overall_winrate_B'].append(wr_b)
        
        # === ОБНОВЛЕНИЕ ПОСЛЕ МАТЧА ===
        win_a = row['winner_is_A']
        
        # Elo update
        exp_a = 1.0 / (1.0 + 10 ** ((team_elo[b] - team_elo[a]) / 400))
        team_elo[a] += K_GLOBAL * (win_a - exp_a)
        team_elo[b] += K_GLOBAL * ((1 - win_a) - (1 - exp_a))
        
        exp_map_a = 1.0 / (1.0 + 10 ** ((team_map_elo[b][m] - team_map_elo[a][m]) / 400))
        team_map_elo[a][m] += K_MAP * (win_a - exp_map_a)
        team_map_elo[b][m] += K_MAP * ((1 - win_a) - (1 - exp_map_a))
        
        # H2H update
        h2h_stats[h2h_key]['total'] += 1
        if (win_a and a < b) or (not win_a and b < a):
            h2h_stats[h2h_key]['wins_A'] += 1
        
        # History update
        team_history[a].append((win_a, team_elo[b], current_date))
        team_history[b].append((1 - win_a, team_elo[a], current_date))
        team_map_history[a][m].append(win_a)
        team_map_history[b][m].append(1 - win_a)
        
        # Date update
        team_last_date[a] = current_date
        team_last_date[b] = current_date
        
        # Streak update
        if win_a:
            team_streak[a] = max(1, team_streak[a] + 1)
            team_streak[b] = min(-1, team_streak[b] - 1) if team_streak[b] <= 0 else -1
        else:
            team_streak[b] = max(1, team_streak[b] + 1)
            team_streak[a] = min(-1, team_streak[a] - 1) if team_streak[a] <= 0 else -1
    
    # Добавляем фичи в датафрейм
    for col, values in features.items():
        df[col] = values
    
    return df


if __name__ == "__main__":
    process_scraped_data()
