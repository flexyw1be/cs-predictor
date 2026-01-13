import pandas as pd
import numpy as np
from collections import defaultdict, deque


def compute_advanced_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    K = 20
    init_elo = 1500
    team_elo = defaultdict(lambda: init_elo)
    team_map_elo = defaultdict(lambda: defaultdict(lambda: init_elo))
    h2h_stats = defaultdict(lambda: {'wins_A': 0, 'total': 0})
    team_map_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=5)))

    elo_diffs, map_elo_diffs, h2h_rates, momentum_diffs = [], [], [], []

    for _, row in df.iterrows():
        a, b, m = row['team_A'], row['team_B'], row['map']

        # Запись текущих показателей
        elo_diffs.append(team_elo[a] - team_elo[b])
        map_elo_diffs.append(team_map_elo[a][m] - team_map_elo[b][m])

        h2h = h2h_stats[tuple(sorted((a, b)))]
        rate = h2h['wins_A'] / h2h['total'] if h2h['total'] > 0 else 0.5
        h2h_rates.append(rate if a < b else 1 - rate)

        m_a = np.mean(team_map_history[a][m]) if team_map_history[a][m] else 0.5
        m_b = np.mean(team_map_history[b][m]) if team_map_history[b][m] else 0.5
        momentum_diffs.append(m_a - m_b)

        # Обновление после матча
        win_a = 1 if row['winner'] == a else 0
        p_a = 1.0 / (1.0 + 10 ** ((team_elo[b] - team_elo[a]) / 400))
        team_elo[a] += K * (win_a - p_a)
        team_elo[b] += K * ((1 - win_a) - (1 - p_a))

        h_key = tuple(sorted((a, b)))
        h2h_stats[h_key]['total'] += 1
        if (win_a and a < b) or (not win_a and b < a):
            h2h_stats[h_key]['wins_A'] += 1

        team_map_history[a][m].append(win_a)
        team_map_history[b][m].append(1 - win_a)

    df['elo_diff'] = elo_diffs
    df['map_elo_diff'] = map_elo_diffs
    df['h2h_rate'] = h2h_rates
    df['momentum_diff'] = momentum_diffs
    return df
