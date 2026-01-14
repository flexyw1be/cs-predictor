import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold


def train_cb_native():
    df_train = pd.read_csv('data/processed/train.csv')
    feature_cols = ['map', 'rank_diff', 'abs_rank_diff', 'picked_by_is_A', 'is_decider',
                    'map_winrate_A', 'map_winrate_B', 'recent_form_A', 'recent_form_B',
                    'elo_diff', 'map_elo_diff']
    X, y = df_train[feature_cols], df_train['winner_is_A']

    model = CatBoostClassifier(cat_features=['map'], logging_level='Silent', random_seed=85)

    # Сетка параметров для поиска
    param_grid = {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'iterations': [100, 500, 1000]
    }

    print("Запуск родного поиска параметров CatBoost (Grid Search)...")
    # Используем встроенный метод CatBoost
    grid_search_result = model.grid_search(param_grid, X=X, y=y, cv=5, stratified=True, plot=False)

    print(f"\nЛучшие параметры: {grid_search_result['params']}")

    if not os.path.exists('models'): os.makedirs('models')
    model.save_model('models/cb_final.bin')
    print("Модель сохранена в models/cb_final.bin")


if __name__ == "__main__":
    train_cb_native()