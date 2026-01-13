import pandas as pd
import joblib


def get_prediction(match_data):
    """
    match_data: dict с ключами как в feature_cols
    Пример: {'map': 'Mirage', 'rank_diff': -10, ...}
    """
    model = joblib.load('models/rf_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')

    df_input = pd.DataFrame([match_data])
    X_proc = preprocessor.transform(df_input)

    prob = model.predict_proba(X_proc)[0, 1]
    return prob


# Пример вызова:
new_match = {
    'map': 'Inferno',
    'rank_diff': -15,
    'abs_rank_diff': 15,
    'picked_by_is_A': 1,
    'is_decider': 0,
    'map_winrate_A': 0.65,
    'map_winrate_B': 0.45,
    'recent_form_A': 0.8,
    'recent_form_B': 0.5,
    'elo_diff': 120,
    'map_elo_diff': 45
}

win_chance = get_prediction(new_match)
print(f"Шанс победы Team A на этой карте: {win_chance:.2%}")
print(f"Шанс победы Team B на этой карте: {1 - win_chance:.2%}")