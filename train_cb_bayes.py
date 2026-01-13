import pandas as pd
import os
import numpy as np
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator, ClassifierMixin


# 1. ОБЕРТКА С ЯВНОЙ ПЕРЕДАЧЕЙ КАТЕГОРИЙ
class CatBoostSklearnWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, iterations=500, depth=6, learning_rate=0.03, cat_features=None):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.cat_features = cat_features
        self.model_ = None

    def get_params(self, deep=True):
        return {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "cat_features": self.cat_features
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def fit(self, X, y, **kwargs):
        # Создаем модель именно здесь, используя актуальные параметры
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            cat_features=self.cat_features,  # Критически важно здесь!
            logging_level='Silent',
            allow_writing_files=False,
            thread_count=-1
        )
        self.model_.fit(X, y, **kwargs)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def save_model(self, path):
        if self.model_:
            self.model_.save_model(path)


def train_cb_bayes():
    # --- ЗАГРУЗКА ---
    train_path = 'data/processed/train.csv'
    if not os.path.exists(train_path):
        print("Ошибка: Сначала запусти data_processor.py")
        return

    df_train = pd.read_csv(train_path)
    feature_cols = [
        'map', 'rank_diff', 'abs_rank_diff', 'picked_by_is_A',
        'is_decider', 'map_winrate_A', 'map_winrate_B',
        'recent_form_A', 'recent_form_B', 'elo_diff', 'map_elo_diff'
    ]
    X, y = df_train[feature_cols], df_train['winner_is_A']

    # --- ОПТИМИЗАЦИЯ ---
    # Указываем, что первая колонка ('map') — категориальная
    base_model = CatBoostSklearnWrapper(cat_features=['map'])

    search_spaces = {
        'iterations': Integer(100, 800),
        'depth': Integer(4, 10),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform')
    }

    print("Запуск Байесовской оптимизации CatBoost (фикс категорий)...")
    opt = BayesSearchCV(
        estimator=base_model,
        search_spaces=search_spaces,
        n_iter=20,  # Для теста 20 итераций
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
        random_state=85,
        verbose=1
    )

    # Передаем группы для GroupKFold
    opt.fit(X, y, groups=df_train['match_id'])

    # --- СОХРАНЕНИЕ ---
    if not os.path.exists('models'):
        os.makedirs('models')

    opt.best_estimator_.save_model('models/cb_bayes.bin')

    print("\n" + "=" * 30)
    print(f"Лучший результат ROC-AUC: {opt.best_score_:.4f}")
    print(f"Параметры: {opt.best_params_}")
    print("=" * 30)


if __name__ == "__main__":
    train_cb_bayes()