import pandas as pd
from config import DROP_COLS, TARGET_COL


def feature_engineering(df):

    df['winrate_diff_calc'] = df['map_winrate_A'] - df['map_winrate_B']
    df['form_diff_calc'] = df['recent_form_A'] - df['recent_form_B']


    df['A_power_index'] = df['recent_form_A'] / (df['team_A_rank'] + 1)
    df['B_power_index'] = df['recent_form_B'] / (df['team_B_rank'] + 1)
    df['power_diff'] = df['A_power_index'] - df['B_power_index']

    y = df[TARGET_COL]

    X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors='ignore')

    X = X.fillna(0)

    print(f"Features engineered. Features count: {X.shape[1]}")
    return X, y