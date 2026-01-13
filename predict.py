import joblib
import pandas as pd
import numpy as np


def get_team_stats(team_name):
    # В идеале здесь должен быть вызов твоего парсера,
    # который берет данные с HLTV (rating, winrate, recent_form)
    # Сейчас используем примерные данные для NAVI и Spirit
    stats_db = {
        "NAVI": {
            "recent_form": 0.85,
            "avg_rating": 1.12,
            "winrate_last_10": 0.80,
            "rank": 1
        },
        "Spirit": {
            "recent_form": 1.90,
            "avg_rating": 2.06,
            "winrate_last_10": 1.60,
            "rank": 2
        },
        "OG":{
            "recent_form": 0.40,
            "avg_rating": 1.06,
            "winrate_last_10": 0.40,
            "rank": 1
        }
    }
    return stats_db.get(team_name)


def predict_match(team_a_name, team_b_name):
    # 1. Загрузка модели и списка признаков (сохраненных GA)
    try:
        data = joblib.load('cs2_predictor_model.joblib')
        model = data['model']
        features = data['features']
    except FileNotFoundError:
        return "Ошибка: Модель не найдена. Обучи её сначала!"

    # 2. Получение данных по командам
    stats_a = get_team_stats(team_a_name)
    stats_b = get_team_stats(team_b_name)

    # 3. Формирование входного вектора (Data Alignment)
    # Очень важно, чтобы признаки шли в том же порядке, что и при обучении
    input_row = {}
    for col in features:
        # Логика маппинга признаков (пример):
        if 'recent_form_A' in col:
            input_row[col] = stats_a['recent_form']
        elif 'recent_form_B' in col:
            input_row[col] = stats_b['recent_form']
        elif 'rating_A' in col:
            input_row[col] = stats_a['avg_rating']
        elif 'rating_B' in col:
            input_row[col] = stats_b['avg_rating']
        else:
            input_row[col] = 0  # Для остальных признаков

    df_match = pd.DataFrame([input_row])[features]

    # 4. Прогноз
    probability = model.predict_proba(df_match)[0]

    print(f"--- АНАЛИЗ МАТЧА: {team_a_name} vs {team_b_name} ---")
    print(f"Вероятность победы {team_a_name}: {probability[1] * 100:.1f}%")
    print(f"Вероятность победы {team_b_name}: {probability[0] * 100:.1f}%")

    # 5. Вердикт на основе уверенности модели
    diff = abs(probability[1] - probability[0])
    if diff > 0.3:
        winner = team_a_name if probability[1] > probability[0] else team_b_name
        print(f"ВЕРДИКТ: Уверенная ставка на {winner}")
    else:
        print("ВЕРДИКТ: Силы равны, матч непредсказуем (Skip)")


if __name__ == "__main__":
    predict_match("OG", "Spirit")

