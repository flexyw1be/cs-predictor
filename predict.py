import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier


def get_prediction(match_data, confidence_threshold=0.5):
    """
    Получить предсказание для матча.
    
    Args:
        match_data: dict с ключами как в feature_cols
        confidence_threshold: минимальный порог уверенности (0.5-0.7)
        
    Returns:
        dict с вероятностью, предсказанием и уровнем уверенности
    """
    # Загрузка модели
    try:
        model = CatBoostClassifier()
        model.load_model('models/catboost_model.cbm')
        feature_cols = joblib.load('models/feature_cols.pkl')
    except:
        # Fallback на старую модель
        model = joblib.load('models/rf_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        df_input = pd.DataFrame([match_data])
        X_proc = preprocessor.transform(df_input[feature_cols])
        prob = model.predict_proba(X_proc)[0, 1]
        return {
            'prob_A': prob,
            'prob_B': 1 - prob,
            'prediction': 'A' if prob > 0.5 else 'B',
            'confident': abs(prob - 0.5) >= (confidence_threshold - 0.5)
        }

    df_input = pd.DataFrame([match_data])
    prob = model.predict_proba(df_input[feature_cols])[0, 1]
    
    # Определяем уровень уверенности
    confidence = abs(prob - 0.5) * 2  # 0 to 1 scale
    is_confident = prob >= confidence_threshold or prob <= (1 - confidence_threshold)
    
    confidence_level = 'low'
    if confidence >= 0.2:
        confidence_level = 'medium'
    if confidence >= 0.3:
        confidence_level = 'high'
    if confidence >= 0.4:
        confidence_level = 'very_high'
    
    return {
        'prob_A': prob,
        'prob_B': 1 - prob,
        'prediction': 'A' if prob > 0.5 else 'B',
        'confidence': confidence,
        'confidence_level': confidence_level,
        'should_bet': is_confident,
        'recommendation': f"{'Рекомендуем ставку' if is_confident else 'Пропустить матч'}"
    }


def print_prediction(result, team_a_name='Team A', team_b_name='Team B'):
    """Красиво выводит результат предсказания"""
    print("=" * 50)
    print(f"{'ПРЕДСКАЗАНИЕ МАТЧА':^50}")
    print("=" * 50)
    print(f"{team_a_name:>20} vs {team_b_name:<20}")
    print("-" * 50)
    print(f"Шанс победы {team_a_name}: {result['prob_A']:>6.1%}")
    print(f"Шанс победы {team_b_name}: {result['prob_B']:>6.1%}")
    print("-" * 50)
    print(f"Предсказание: {team_a_name if result['prediction'] == 'A' else team_b_name}")
    print(f"Уровень уверенности: {result['confidence_level']}")
    print(f"Рекомендация: {result['recommendation']}")
    print("=" * 50)


# Пример вызова:
if __name__ == "__main__":
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
        'map_elo_diff': 45,
        'h2h_rate': 0.7,
        'h2h_games': 5,
        'momentum_diff': 0.2,
        'streak_A': 3,
        'streak_B': -1,
        'days_since_last_A': 2,
        'days_since_last_B': 5,
        'overall_winrate_A': 0.7,
        'overall_winrate_B': 0.5,
        'winrate_diff': 0.2,
        'opponent_strength_A': 1520,
        'opponent_strength_B': 1480,
        'map_games_A': 15,
        'map_games_B': 10,
        'map_experience_diff': 5,
    }

    result = get_prediction(new_match, confidence_threshold=0.60)
    print_prediction(result, "NaVi", "G2")