import pandas as pd
import numpy as np
from collections import defaultdict, deque


def compute_advanced_features(df):
    """
    Вычисляет продвинутые признаки для предсказания результатов матчей CS.
    
    Добавляемые фичи:
    - elo_diff: разница общего Elo рейтинга
    - map_elo_diff: разница Elo по конкретной карте
    - h2h_rate: история личных встреч
    - momentum_diff: разница моментума на карте
    - streak_A/B: текущая серия побед/поражений
    - days_since_last_A/B: дней с последнего матча
    - overall_winrate_A/B: общий винрейт за последние N игр
    - opponent_strength_A/B: средняя сила последних соперников
    - map_games_A/B: количество сыгранных игр на карте
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Параметры Elo
    K_GLOBAL = 32  # K-фактор для общего Elo
    K_MAP = 40     # K-фактор для Elo по карте (выше, т.к. меньше игр)
    INIT_ELO = 1500
    HISTORY_SIZE = 10  # Размер окна для скользящих статистик
    
    # Хранилища состояния
    team_elo = defaultdict(lambda: INIT_ELO)
    team_map_elo = defaultdict(lambda: defaultdict(lambda: INIT_ELO))
    h2h_stats = defaultdict(lambda: {'wins_A': 0, 'total': 0})
    
    # История матчей для скользящих статистик
    team_history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))  # (win, opponent_elo, date)
    team_map_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=HISTORY_SIZE)))
    team_last_date = {}
    team_streak = defaultdict(int)  # положительное = серия побед, отрицательное = поражений
    team_map_games = defaultdict(lambda: defaultdict(int))
    
    # Списки для результатов
    features = {
        'elo_diff': [],
        'map_elo_diff': [],
        'h2h_rate': [],
        'h2h_games': [],
        'momentum_diff': [],
        'streak_A': [],
        'streak_B': [],
        'days_since_last_A': [],
        'days_since_last_B': [],
        'overall_winrate_A': [],
        'overall_winrate_B': [],
        'winrate_diff': [],
        'opponent_strength_A': [],
        'opponent_strength_B': [],
        'map_games_A': [],
        'map_games_B': [],
        'map_experience_diff': [],
    }

    for _, row in df.iterrows():
        a, b, m = row['team_A'], row['team_B'], row['map']
        current_date = row['date']

        # === ЗАПИСЬ ТЕКУЩИХ ПОКАЗАТЕЛЕЙ (до обновления!) ===
        
        # Elo разницы
        features['elo_diff'].append(team_elo[a] - team_elo[b])
        features['map_elo_diff'].append(team_map_elo[a][m] - team_map_elo[b][m])
        
        # H2H статистика
        h2h_key = tuple(sorted((a, b)))
        h2h = h2h_stats[h2h_key]
        if h2h['total'] > 0:
            rate = h2h['wins_A'] / h2h['total']
            h2h_rate = rate if a < b else 1 - rate
        else:
            h2h_rate = 0.5
        features['h2h_rate'].append(h2h_rate)
        features['h2h_games'].append(h2h['total'])
        
        # Momentum на карте (скользящий винрейт)
        hist_a = team_map_history[a][m]
        hist_b = team_map_history[b][m]
        m_a = np.mean(hist_a) if hist_a else 0.5
        m_b = np.mean(hist_b) if hist_b else 0.5
        features['momentum_diff'].append(m_a - m_b)
        
        # Streak
        features['streak_A'].append(team_streak[a])
        features['streak_B'].append(team_streak[b])
        
        # Дни с последнего матча
        if a in team_last_date:
            days_a = (current_date - team_last_date[a]).days
        else:
            days_a = 30  # default для новых команд
        if b in team_last_date:
            days_b = (current_date - team_last_date[b]).days
        else:
            days_b = 30
        features['days_since_last_A'].append(days_a)
        features['days_since_last_B'].append(days_b)
        
        # Общий винрейт из последних N игр
        if team_history[a]:
            wr_a = np.mean([h[0] for h in team_history[a]])
        else:
            wr_a = 0.5
        if team_history[b]:
            wr_b = np.mean([h[0] for h in team_history[b]])
        else:
            wr_b = 0.5
        features['overall_winrate_A'].append(wr_a)
        features['overall_winrate_B'].append(wr_b)
        features['winrate_diff'].append(wr_a - wr_b)
        
        # Средняя сила соперников (opponent strength)
        if team_history[a]:
            opp_str_a = np.mean([h[1] for h in team_history[a]])
        else:
            opp_str_a = INIT_ELO
        if team_history[b]:
            opp_str_b = np.mean([h[1] for h in team_history[b]])
        else:
            opp_str_b = INIT_ELO
        features['opponent_strength_A'].append(opp_str_a)
        features['opponent_strength_B'].append(opp_str_b)
        
        # Опыт на карте
        features['map_games_A'].append(team_map_games[a][m])
        features['map_games_B'].append(team_map_games[b][m])
        features['map_experience_diff'].append(team_map_games[a][m] - team_map_games[b][m])

        # === ОБНОВЛЕНИЕ ПОСЛЕ МАТЧА ===
        win_a = 1 if row['winner'] == a else 0
        
        # Обновление общего Elo
        expected_a = 1.0 / (1.0 + 10 ** ((team_elo[b] - team_elo[a]) / 400))
        team_elo[a] += K_GLOBAL * (win_a - expected_a)
        team_elo[b] += K_GLOBAL * ((1 - win_a) - (1 - expected_a))
        
        # Обновление Elo по карте (ИСПРАВЛЕННЫЙ БАГ!)
        expected_map_a = 1.0 / (1.0 + 10 ** ((team_map_elo[b][m] - team_map_elo[a][m]) / 400))
        team_map_elo[a][m] += K_MAP * (win_a - expected_map_a)
        team_map_elo[b][m] += K_MAP * ((1 - win_a) - (1 - expected_map_a))
        
        # Обновление H2H
        h2h_stats[h2h_key]['total'] += 1
        if (win_a and a < b) or (not win_a and b < a):
            h2h_stats[h2h_key]['wins_A'] += 1
        
        # Обновление истории матчей
        team_history[a].append((win_a, team_elo[b], current_date))
        team_history[b].append((1 - win_a, team_elo[a], current_date))
        
        # Обновление истории на карте
        team_map_history[a][m].append(win_a)
        team_map_history[b][m].append(1 - win_a)
        
        # Обновление даты последнего матча
        team_last_date[a] = current_date
        team_last_date[b] = current_date
        
        # Обновление streak
        if win_a:
            team_streak[a] = max(1, team_streak[a] + 1)
            team_streak[b] = min(-1, team_streak[b] - 1) if team_streak[b] <= 0 else -1
        else:
            team_streak[b] = max(1, team_streak[b] + 1)
            team_streak[a] = min(-1, team_streak[a] - 1) if team_streak[a] <= 0 else -1
        
        # Обновление счётчика игр на карте
        team_map_games[a][m] += 1
        team_map_games[b][m] += 1

    # Добавляем все фичи в датафрейм
    for col, values in features.items():
        df[col] = values
    
    return df
