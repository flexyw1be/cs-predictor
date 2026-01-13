import pandas as pd
from config import DROP_COLS, TARGET_COL


def feature_engineering(df):
    df['winrate_diff'] = df['map_winrate_A'] - df['map_winrate_B']
    df['abs_winrate_diff'] = abs(df['map_winrate_A'] - df['map_winrate_B'])

    df['form_diff'] = df['recent_form_A'] - df['recent_form_B']
    df['abs_form_diff'] = abs(df['recent_form_A'] - df['recent_form_B'])

    df['A_power_index'] = df['recent_form_A'] / (df['team_A_rank'] + 1)
    df['B_power_index'] = df['recent_form_B'] / (df['team_B_rank'] + 1)

    df['power_diff'] = df['A_power_index'] - df['B_power_index']
    df['abs_power_diff'] = abs(df['A_power_index'] - df['B_power_index'])
    cols_to_drop = [
        'A_power_index', 'B_power_index',
        'team_A_rank', 'team_B_rank',
        'avg_rating_A', 'avg_rating_B',
        'winrate_A', 'winrate_B',
        'recent_form_A', 'recent_form_B',
        'map'
    ]

    # Удаляем только если они есть в датафрейме
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    if 'rating_diff' not in df.columns and 'avg_rating_A' in df.columns:
        df['rating_diff'] = df['avg_rating_A'] - df['avg_rating_B']

    print(f"--- Оптимизация признаков: удалено {len(existing_drops)} колонок ---")
    print(f"Оставшиеся признаки: {list(df.columns)}")

    y = df[TARGET_COL]

    X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors='ignore')

    X = X.fillna(0)

    print(f"Features engineered. Features count: {X.shape[1]}")
    return X, y
